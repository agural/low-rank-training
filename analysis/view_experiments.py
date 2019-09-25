import os, sys, pdb, time, pickle
import numpy as np
#from legacy.mnist import *

last_update = 0
while True:
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    fnames = os.listdir(output_dir)
    fnames.sort(key=lambda s: os.path.getmtime(os.path.join(output_dir, s)))
    cur_update = os.path.getmtime(os.path.join(output_dir, fnames[-1]))
    if cur_update <= last_update:
        time.sleep(1)
        continue
    last_update = cur_update
    print()
    print('='*80)
    for i in range(10): print()
    print('='*80)
    os.system('cls' if os.name == 'nt' else 'clear')
    for fname in fnames:
        if fname[-6:] != 'meta.p':
            continue
        try:
            with open(os.path.join(output_dir, fname), 'rb') as f:
                meta = pickle.load(f)
        except Exception as e:
            print('%s: Error - %s'%(fname, e))
            continue
        TrAccs = ['%.4f'%x['TrAcc'] for x in meta['Train']]
        TeAccs = ['%.4f'%(x.get('TeAcc') or 0) for x in meta['Train']]
        loss = meta['Train'][-1].get('TeLoss') or 0
        acc  = meta['Train'][-1].get('TeAcc') or 0
        if meta['Test'] is not None:
            loss = meta['Test']['TeLoss']
            acc  = meta['Test']['TeAcc']
        density = meta['Train'][-1].get('UpdateDensity') or np.nan
        print('\n%s: %s'%(meta['Name'], meta['Args'].description))
        print('\tTrAccs: %s'%(' '.join(TrAccs)))
        print('\tTeAccs: %s'%(' '.join(TeAccs)))
        print('\tEpochs: %03d - TeLoss: %6s - TeAcc: %.4f - Density: %.4e'%(
            len(meta['Train']), '%.4f'%loss, acc, density))
    time.sleep(10)

