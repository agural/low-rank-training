import os, sys, pdb, pickle, pathlib, argparse
from profilehooks import profile
from collections import OrderedDict

import copy, time, math, random
import numpy as np
import scipy as sp
from scipy.spatial.distance import cosine

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import torch

import definitions
import data.data_loader as data
from pytorch.main import NetPT

from lr.utils import *
from lr.layers import *
from lr.net import *
from lr.experiments import configs


outroot = os.path.join(definitions.ROOT_DIR, 'analysis')
train_meta = {}
best_acc = 0.0

def train(model, train_loader, data_meta, args, epoch, model0):
    global train_meta
    conf = configs[args.name]
    model.set_mode(train=True)
    tr_loss = 0.0
    tr_acc = 0.0
    tr_loss_mv = 0.0
    tr_acc_mv = 0.0
    n_samp = 0
    n_batches = len(train_loader)
    n_classes = data_meta['output_shape'][-1]
    tr_losses = np.zeros(n_batches, dtype='f4')
    tr_acces  = np.zeros(n_batches, dtype='f4')
    updmh = np.zeros(n_batches, dtype='i4')
    t0 = time.time()
    checkQs = []
    model_summaries = []
    for batch_idx, (data, target) in enumerate(train_loader):
        bs = len(target)
        X = data.numpy().astype(dt)
        yt = target.numpy().flatten()
        Yt = np.zeros((bs, 10))
        Yt[np.arange(bs),yt] = 1

        drift_downsample = 10 # Downsampling is only used to speed up training.
        if batch_idx % drift_downsample == 0 and conf.add_drift is not None:
            model.drift(conf.add_drift, drift_downsample)

        Y = model(X)
        loss = model.loss_fn(Y, Yt)
        if np.isnan(loss):
            print('train')
            pdb.set_trace()
        if conf.train:
            model.backward()
            model.update(conf.lr)

        updd = model.get_update_density()
        updm = model.get_worst_case_updates()
        updmh[batch_idx] = updm
        rel_errors = sorted([(q.rel_error, p) for p,q in model.get_all(FixedQuantize, '')])

        yp = np.argmax(Y, 1)
        correct = np.sum(yp == yt)
        tr_acc  += correct
        tr_loss += bs * loss
        n_samp  += bs
        tr_acc_mv  = 0.99 * tr_acc_mv + 0.01 * (correct / bs)
        tr_loss_mv = 0.99 * tr_loss_mv + 0.01 * loss
        tr_acces[batch_idx]  = correct/bs
        tr_losses[batch_idx] = loss
        track = model_summaries[-1]['conv1/W']['cos'] if len(model_summaries) else 0
        
        last = (batch_idx == len(train_loader) - 1)
        if batch_idx % 100 == 0 or last:
            model_summaries.append(get_summary(model0, model))
        train_meta = {
            'Time':time.time()-t0,
            'LR':conf.lr,
            'UpdateDensity':updd,
            'UpdateMaxHist':updmh[:batch_idx+1],
            'TrLoss':tr_loss/n_samp,
            'TrAcc':tr_acc/n_samp,
            'TrLossHist':tr_losses[:batch_idx+1],
            'TrAccHist':tr_acces[:batch_idx+1],
            'QRelErr':rel_errors,
            'Distributions':model_summaries,
            'UHist':model.uhist(),
        }
        print('\rEpoch: %03d/%03d - Step: %05d/%05d (%5.1f%%) - '
                'Time: %06.1fs - UpdMax: %.3e - Track: %.3e - MaxQRelErr: %.4f (%6s) - TrLoss: %.4f(%.4f) - TrAcc: %.4f(%.4f)'%(
            epoch, conf.epochs, batch_idx+1, n_batches,
            100. * (batch_idx+1) / len(train_loader), time.time() - t0, updm,
            track,
            rel_errors[-1][0], (' '*6 + rel_errors[-1][1])[-6:],
            tr_loss / n_samp, tr_loss_mv, tr_acc / n_samp, tr_acc_mv), end='')
    return train_meta

def test(model, test_loader, data_meta, verbose=True):
    model.set_mode(train=False)
    global best_acc
    test_loss = 0
    correct = 0
    n_batches = len(test_loader)
    n_classes = data_meta['output_shape'][-1]
    for batch_idx, (data, target) in enumerate(test_loader):
        bs = len(target)
        X = data.numpy().astype(dt)
        yt = target.numpy().flatten()
        Yt = np.zeros((bs, n_classes))
        Yt[np.arange(bs),yt] = 1
        Y = model(X)
        test_loss += bs * model.loss_fn(Y, Yt)
        yp = np.argmax(Y, 1)
        correct += np.sum(yp == yt)
        if verbose:
            progress = ' - %d/%d'%(batch_idx+1, n_batches)
            print(progress + '\b'*len(progress), end='')
            sys.stdout.flush()
    test_acc  = correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    if verbose:
        print(' - TeLoss: %.4f - TeAcc: %.4f'%(test_loss, test_acc), end='')
        if best_acc < test_acc:
            best_acc = test_acc
            print(' *')
        else:
            print('')
    meta = {
        'TeLoss':test_loss,
        'TeAcc':test_acc
        }
    return meta

def init_from_pytorch(model, model_fname, valid_loader, data_meta, verify=2):
    pt_model = torch.load(model_fname, map_location='cpu')
    model.init_from_pytorch(pt_model)
    if verify in [1, 3]:
        Xsamp_pt, ysamp_pt = next(iter(valid_loader))
        Xsamp = Xsamp_pt.numpy(); ysamp = ysamp_pt.numpy()
        model.set_mode(train=False)
        model(Xsamp)
        pt_model.eval()
        pt_model(Xsamp_pt)
        def compare(act_name):
            ma = getattr(model, act_name).flatten()
            pa = getattr(pt_model, act_name).detach().numpy().flatten()
            diff = ma - pa
            issues = np.where(np.abs(diff) > 0)[0]
            if len(issues) == 0:
                iss = []
            else:
                iw = np.argmax(np.abs(diff))
                iss = diff[iw:iw+5]
            print('Comparing %s - %8.4f%% issues, eg'%(
                act_name, 100.0*len(issues)/diff.size), iss)
            return ma, pa, issues
        compare('x1')
        compare('x2')
        compare('x4')
        compare('x5')
        compare('x6')
        compare('x8')
        compare('x9')
        compare('x10')
        compare('x11')
        compare('x12')
        model.set_mode(train=True)
    if verify in [2, 3]:
        print('Initialized to %s. Checking validation accuracy: '%(
            model_fname.split('/')[-1]), end='')
        test(model, valid_loader, data_meta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='lrt-base', help='Experiment name (default lrt-base).')
    parser.add_argument('--save_model', default=1, type=int, help='Whether to pickle/save the model.')
    parser.add_argument('--pdb', default=1, type=int, help='Whether to end with pdb debugger.')
    args = parser.parse_args()
    conf = configs[args.name]
    args.description = conf.description

    # Prepare randomness for reproducibility.
    if conf.seed is not None:
        print('Using deterministic seed: %d'%conf.seed)
        random.seed(conf.seed)
        np.random.seed(conf.seed)
        torch.manual_seed(conf.seed)
        torch.backends.cudnn.deterministic = True

    # Prepare files for saving.
    name = args.name
    output_dir = os.path.join(outroot, 'output')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    meta_fname = os.path.join(output_dir, '%s-meta.p'%name)
    ckpt_files_content = definitions.get_ckpt_files()

    # Get data.
    data_loader = data.get[conf.dataset]
    train_loader, valid_loader, test_loader = data_loader()
    data_sample = next(iter(test_loader))
    data_samplex = data_sample[0].cpu().numpy()
    data_sampley = data_sample[1].cpu().numpy()
    try:
        data_classes = test_loader.dataset.classes
    except Exception as e:
        data_classes = [str(x) for x in range(max(data_sampley) + 1)]
    data_meta = OrderedDict([
        ('name',conf.dataset),
        ('input_shape',data_samplex.shape[1:]),
        ('output_shape',(len(data_classes),)),
        ('input_range',[np.min(data_samplex), np.max(data_samplex)]),
        ('output_labels', data_classes),
    ])
    print('Dataset details: %s'%conf.dataset)
    for k,v in data_meta.items():
        print('\t%s:'%k, v)
    print('='*80)
    print('Configuration details:')
    print('\t' + str(conf).replace('\n', '\n\t'))
    print('='*80)

    # Load model.
    model = Net(conf, data_meta)
    model.set_mode(quant=True, qcal=False)
    if conf.train_1ep_fp:
        model.set_mode(quant=False, qcal=False)
    if conf.pt_init is not None:
        model_fname = os.path.join(output_dir, conf.pt_init)
        init_from_pytorch(model, model_fname, valid_loader, data_meta, verify=1)
    model0 = copy.deepcopy(model)

    # Run training.
    t0 = time.time()
    meta = {'Args':args, 'Name':name, 'Description':conf.description, 'Config':conf.ser(),
            'Files':ckpt_files_content, 'Start':t0, 'Train':[], 'Test':None}
    try:
        for epoch in range(1, conf.epochs+1):
            meta_epoch = {'Epoch':epoch}
            mtr = train(model, train_loader, data_meta, args, epoch, model0)
            meta_epoch.update(mtr)
            if conf.validate:
                if conf.train_1ep_fp:
                    model.set_mode(qcal=True)
                mte = test(model, valid_loader, data_meta)
                if conf.train_1ep_fp:
                    model.set_mode(quant=True, qcal=False)
                meta_epoch.update(mte)
            meta['Train'].append(meta_epoch)
            with open(meta_fname, 'wb') as f:
                pickle.dump(meta, f)
            if args.save_model:
                with open(os.path.join(outroot, 'models', '%s.p'%name), 'wb') as f:
                    pickle.dump(model, f)
    except KeyboardInterrupt:
        print()
        mtr = train_meta
        meta_epoch.update(mtr)
        mte = {'TeLoss':np.nan, 'TeAcc':np.nan}
        meta_epoch.update(mte)
        meta['Train'].append(meta_epoch)
        meta['Test'] = {'TeLoss':np.nan, 'TeAcc':np.nan, 'Time':time.time()-t0}
        print('Starting debugger. Enter "c" to save meta file.')
        pdb.set_trace()
        with open(meta_fname, 'wb') as f:
            pickle.dump(meta, f)
        print('Starting debugger. Enter "c" to run final test set before exiting.')
        pdb.set_trace()

    if conf.validate:
        mte = test(model, test_loader, data_meta)
        mte['Time'] = time.time() - t0
        meta['Test'] = mte
        print('Final Test Loss: %.4f - Test Accuracy: %.4f'%(
            mte['TeLoss'], mte['TeAcc']))
    with open(meta_fname, 'wb') as f:
        pickle.dump(meta, f)

    print('Training complete. Starting debugger.')
    if args.pdb:
        pdb.set_trace()
        
if __name__ == '__main__':
    main()

