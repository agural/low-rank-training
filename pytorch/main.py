import os, sys, pdb, gc, pickle, pathlib, argparse
from collections import OrderedDict

import time, math, random
import numpy as np
import scipy as sp

import torch
import torch.nn.functional as F
import torch.nn as nn

import definitions
import data.data_loader as data
from pytorch.utils import *
from pytorch.layers import *
from pytorch.net import *

outroot = os.path.join(definitions.ROOT_DIR, 'analysis')

train_meta = {}
def train(model, obj, opt, qopt, train_loader, data_meta, args, epoch):
    global train_meta
    model.train()
    tr_loss = 0.0
    tr_acc = 0.0
    n_samp = 0
    n_batches = len(train_loader)
    n_classes = data_meta['output_shape'][-1]
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        bs = len(target)
        X = data.to(device)
        yt = target.to(device).view(-1)
        Yp = model(X)
        
        loss = obj(Yp, yt)
        if np.isnan(loss.item()) or loss.item() > 5:
            print('train')
            pdb.set_trace()
        opt.zero_grad()
        if qopt: qopt.zero_grad()
        loss.backward()
        opt.step()
        if qopt: qopt.step()
        model.force_quantize()

        tr_acc  += np.sum((Yp.max(dim=1)[1] == yt).cpu().numpy())
        tr_loss += bs * loss.item()
        n_samp  += bs
        print('\rEpoch: %03d/%03d - Step: %05d/%05d (%5.1f%%) - '
                'Time: %06.1fs - TrLoss: %.4f - TrAcc: %.4f'%(
            epoch, args.epochs, batch_idx+1, n_batches,
            100. * (batch_idx+1) / len(train_loader), time.time() - t0,
            tr_loss / n_samp, tr_acc / n_samp), end='')
        sys.stdout.flush()
        train_meta = {
            'Time':time.time()-t0,
            'LR':args.lr,
            'TrLoss':tr_loss/n_samp,
            'TrAcc':tr_acc/n_samp,
        }
    return train_meta

best_acc = 0.0
def test(model, obj, test_loader, data_meta, verbose=True):
    model.eval()
    global best_acc
    test_loss = 0
    correct = 0
    n_batches = len(test_loader)
    n_classes = data_meta['output_shape'][-1]
    for batch_idx, (data, target) in enumerate(test_loader):
        bs = len(target)
        X = data.to(device)
        yt = target.to(device).view(-1)
        Yp = model(X)
        loss = obj(Yp, yt)
        if loss.item() > 5:
            pdb.set_trace()
        test_loss += bs * loss.item()
        correct += np.sum((Yp.max(dim=1)[1] == yt).cpu().numpy())
        if verbose:
            progress = ' - %d/%d'%(batch_idx+1, n_batches)
            print(progress + '\b'*len(progress), end='')
            sys.stdout.flush()
    test_acc  = correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    best = best_acc < test_acc
    if verbose:
        print(' - TeLoss: %.4f - TeAcc: %.4f'%(test_loss, test_acc), end='')
        if best:
            best_acc = test_acc
            print(' *')
        else:
            print('')
    meta = {
        'TeLoss':test_loss,
        'TeAcc':test_acc,
        'Best':best,
        }
    return meta


def main():
    global best_acc

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='temp', help='Experiment name (default temp)')
    parser.add_argument('--description', default='', help='Brief description of experiment')
    parser.add_argument('--dataset', default='MNIST', help='Name of dataset (MNIST, CIFAR10, SVHN)')
    parser.add_argument('--save_model', default='MNIST_model.pt', help='Name of model save file or None.')
    parser.add_argument('--save_W', default=0, type=int, help='Whether to save before/after models to compare weights.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed (None is non-deterministic).')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs.')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size.')
    parser.add_argument('--lr', default=1e-5 * 256, type=float, help='Learning rate.')
    parser.add_argument('--opt', default='Adam', help='Optimizer (Adam or MaxNorm).')
    parser.add_argument('--quantize_after', default=0, type=int, help='At which epoch the model should be quantized.')
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # Prepare randomness for reproducibility.
    if args.seed is not None:
        print('Using deterministic seed: %d'%args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Prepare files for saving.
    name = args.name
    output_dir = os.path.join(outroot, 'output')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    meta_fname = os.path.join(output_dir, '%s-meta.p'%name)
    ckpt_files_content = definitions.get_ckpt_files()

    # Get data.
    data_loader = data.get[args.dataset]
    train_loader, valid_loader, test_loader = data_loader(train_batch_size=batch_size)
    data_sample = next(iter(test_loader))
    data_samplex = data_sample[0].cpu().numpy()
    data_sampley = data_sample[1].cpu().numpy()
    try:
        data_classes = test_loader.dataset.classes
    except Exception as e:
        data_classes = [str(x) for x in range(max(data_sampley) + 1)]
    data_meta = OrderedDict([
        ('name',args.dataset),
        ('input_shape',data_samplex.shape[1:]),
        ('output_shape',(len(data_classes),)),
        ('input_range',[np.min(data_samplex), np.max(data_samplex)]),
        ('output_labels', data_classes),
    ])
    print('Dataset details: %s'%args.dataset)
    for k,v in data_meta.items():
        print('\t%s:'%k, v)
    print('='*80)

    # Run training.
    t0 = time.time()
    meta = {'Args':args, 'Name':name, 'Description':args.description,
            'Files':ckpt_files_content, 'Start':t0, 'Train':[], 'Test':None}
    qbits = {'w':8, 'b':16, 'a':8, 'wmax':1.0, 'bmax':8.0, 'amax':2.0}
    model = NetPT(data_meta, qbits=qbits, bs=args.batch_size).to(device)

    wd = 0
    obj = nn.CrossEntropyLoss(reduction='mean')
    if args.opt == 'MaxNorm':
        opt = MaxNormOpt(model.parameters(), lr=lr, weight_decay=wd, qbits=qbits['a'])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 0.98 ** epoch)
    qopt = None

    # Calibrate then set to quantization mode.
    #model.set_qmode('C')
    #test(model, obj, valid_loader, data_meta)
    model.set_qmode('F')
    if args.quantize_after == 0:
        model.set_qmode('T')
    if args.save_W:
        torch.save(model, os.path.join(outroot, 'checkW', '%s-init.pt'%args.name))

    try:
        for epoch in range(1, epochs+1):
            scheduler.step()
            meta_epoch = {'Epoch':epoch}
            mtr = train(model, obj, opt, qopt, train_loader, data_meta, args, epoch)
            meta_epoch.update(mtr)
            if args.quantize_after == epoch:
                model.set_qmode('C')
            mte = test(model, obj, valid_loader, data_meta)
            if args.quantize_after == epoch:
                model.set_qmode('T')
                best_acc = 0.0 # Reset so that we pick out one of the quantized models.
            meta_epoch.update(mte)
            meta['Train'].append(meta_epoch)
            if mte['Best'] and args.save_model:
                model_fname = os.path.join(output_dir, args.save_model)
                torch.save(model, model_fname)
            if mte['Best'] and args.save_W:
                torch.save(model, os.path.join(outroot, 'checkW', '%s-trained.pt'%args.name))
            while True:
                try:
                    with open(meta_fname, 'wb') as f:
                        pickle.dump(meta, f)
                    break
                except Exception as e:
                    time.sleep(60)
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
    mte = test(model, obj, test_loader, data_meta)
    mte['Time'] = time.time() - t0
    meta['Test'] = mte
    print('Final Test Loss: %.4f - Test Accuracy: %.4f'%(
        mte['TeLoss'], mte['TeAcc']))
    with open(meta_fname, 'wb') as f:
        pickle.dump(meta, f)

    print('Training complete. Starting debugger.')
    pdb.set_trace()


if __name__ == '__main__':
    main()

