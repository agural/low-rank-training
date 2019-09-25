import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_ckpt_files():
    ckpt_files = {}
    for r, ds, fs in os.walk(ROOT_DIR, topdown=True):
        for reject in ['LR_train.egg-info', '.git']:
            if reject in ds: ds.remove(reject)
        for fname in fs:
            if fname.split('.')[-1] in ['py', 'sh', 'txt', 'md']:
                key = os.path.join(r, fname)[len(ROOT_DIR):]
                ckpt_files[key] = open(os.path.join(r, fname), 'r').read()
    return ckpt_files

