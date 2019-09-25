import os, sys, pdb, h5py, pickle, inspect
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

import definitions

data_root = os.path.join(definitions.ROOT_DIR, 'data')


### Standard Datasets.

def get_loaders(dataset, train_batch_size=1, test_batch_size=128,
        random_seed=0, tfs=[], aug_tfs=[], valid_frac=0.2, shuffle=True, chw_transpose=True):
    '''
    Utility function for getting data loaders for given dataset.
    Modified from: https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123

    Params
    ------
    - dataset: torchvision dataset object.
    - data_dir: path directory to the dataset.
    - train_batch_size: samples per training batch.
    - test_batch_size: samples per validation/testing batch.
    - random_seed: fix seed for reproducibility.
    - tfs: transforms applied to all datasets.
    - aug_tfs: additional transforms applied to augment training set.
    - valid_frac: fraction [0,1] split for val/train. If None, will use test set for validation.
    - shuffle: whether to shuffle the indices.
    - chw_transpose: whether to transpose chw to hwc.

    Returns
    -------
    - (train_loader, valid_loader, test_loader)
    '''

    # Define the transforms.
    transpose = [transforms.Lambda(lambda x: np.transpose(x, (1,2,0)))] if chw_transpose else []
    train_transforms = transforms.Compose(aug_tfs + tfs + [transforms.ToTensor()] + transpose)
    test_transforms = transforms.Compose(tfs + [transforms.ToTensor()] + transpose)

    # Load the dataset.
    if 'train' in inspect.getargspec(dataset).args:
        train_select = {'train':True}
        valid_select = {'train':True}
        test_select  = {'train':False}
    else:
        train_select = {'split':'train'}
        valid_select = {'split':'train'}
        test_select = {'split':'test'}
    train_dataset = dataset(root=data_root, **train_select, download=True, transform=train_transforms)
    valid_dataset = dataset(root=data_root, **valid_select, download=True, transform=test_transforms)
    test_dataset = dataset(root=data_root, **test_select, download=True, transform=test_transforms)
    
    # Split the training set to train/valid.
    if valid_frac is None:
        valid_dataset = test_dataset
    else:
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(valid_frac * num_train)
        if shuffle:
            tmp = np.random.get_state()
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            np.random.set_state(tmp)
        train_idx, valid_idx = indices[split:], indices[:split]
        train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        valid_dataset = torch.utils.data.Subset(valid_dataset, valid_idx)

    # Generate the loaders.
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=shuffle)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=test_batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=test_batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader

def get_mnist(**kwargs):
    return get_loaders(datasets.MNIST, valid_frac=1/6, **kwargs)

def get_fashionmnist(**kwargs):
    return get_loaders(datasets.FashionMNIST, valid_frac=1/6, **kwargs)

def get_cifar10(**kwargs):
    aug = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=[32,32], padding=4),
    ]
    return get_loaders(datasets.CIFAR10, valid_frac=1/5, aug_tfs=aug, **kwargs)

def get_svhn(**kwargs):
    return get_loaders(datasets.SVHN, valid_frac=0.1, **kwargs)


### MNIST Augmented v1

class HDF5Dataset(data.Dataset):
    def __init__(self, root, train=True, download=False, transform=None):
        self.file_path = os.path.join(root, self.file_path)
        self.f = h5py.File(self.file_path, 'r')
        self.train = train
        t = '' if train else 't'
        self.X = self.f['X%s'%t]
        self.y = self.f['y%s'%t]
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        y = self.y[index]
        y = torch.tensor(np.int64(y))
        return (x, y)

    def __len__(self):
        return self.y.shape[0]

class MNIST_aug500k_dataset(HDF5Dataset):
    def __init__(self, train=True, **kwargs):
        self.file_path = 'MNIST-aug500k/mnist-aug500k.hdf5'
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        super(MNIST_aug500k_dataset, self).__init__(train=train, **kwargs)

class MNIST_aug500k_ds_dataset(HDF5Dataset):
    def __init__(self, train=True, **kwargs):
        self.file_path = 'MNIST-aug500k/mnist-aug500k-DS.hdf5'
        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        super(MNIST_aug500k_ds_dataset, self).__init__(train=train, **kwargs)

class MNIST_aug50k_ds_dataset(MNIST_aug500k_ds_dataset):
    def __len__(self):
        return 50000 if self.train else super(MNIST_aug50k_ds_dataset, self).__len__()

def get_mnist_aug500k(**kwargs):
    ds = MNIST_aug500k_dataset
    return get_loaders(ds, valid_frac=None, shuffle=False, **kwargs)

def get_mnist_aug500k_ds(**kwargs):
    ds = MNIST_aug500k_ds_dataset
    return get_loaders(ds, valid_frac=None, shuffle=False, **kwargs)

def get_mnist_aug50k_ds(**kwargs):
    ds = MNIST_aug50k_ds_dataset
    return get_loaders(ds, valid_frac=0.2, shuffle=True, **kwargs)


### MNIST Augmented v2

class MNISTAugDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, sample=None):
        self.sample = sample
        if split == 'train':
            fname = os.path.join(root, 'mnist-aug500k.hdf5')
            self.f = h5py.File(fname, 'r')
            self.X = self.f['Xtr']
            self.y = self.f['ytr']
        if split == 'valid':
            fname = os.path.join(root, 'mnist-aug500k.hdf5')
            self.f = h5py.File(fname, 'r')
            self.X = self.f['Xva']
            self.y = self.f['yva']
        if split == 'test':
            fname = os.path.join(root, 'mnist-aug500k.hdf5')
            self.f = h5py.File(fname, 'r')
            self.X = self.f['Xte']
            self.y = self.f['yte']
        if split == 'test_raw':
            fname = os.path.join(root, 'mnist-aug500k.hdf5')
            self.f = h5py.File(fname, 'r')
            self.X = self.f['Xt']
            self.y = self.f['yt']
        if split == 'online':
            fname = os.path.join(root, 'mnist-aug500k-ON2.hdf5')
            self.f = h5py.File(fname, 'r')
            self.X = self.f['X']
            self.y = self.f['y']
        if split == 'shift':
            fname = os.path.join(root, 'mnist-aug500k-DS2.hdf5')
            self.f = h5py.File(fname, 'r')
            self.X = self.f['X']
            self.y = self.f['y']
        if split == 'shift2':
            fname = os.path.join(root, 'mnist-aug500k-DS2.hdf5')
            self.f = h5py.File(fname, 'r')
            self.X = self.f['X']
            self.y = self.f['y']
        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        y = self.y[index]
        y = torch.tensor(np.int64(y))
        return (x, y)

    def __len__(self):
        if self.sample:
            return min(self.y.shape[0], self.sample)
        else:
            return self.y.shape[0]

def get_hdf5_loader(dataset, ds_root, splits, sample=None):
    transpose = transforms.Lambda(lambda x: np.transpose(x, (1,2,0)))
    all_transforms = transforms.Compose([transforms.ToTensor(), transpose])
    loaders = []
    for split, bs, shuffle in splits:
        ds_init = dataset(root=ds_root, split=split, transform=all_transforms, sample=sample)
        loader = torch.utils.data.DataLoader(ds_init,
            batch_size=bs, shuffle=shuffle)
        loaders.append(loader)
    return loaders

def get_mnist_aug_off(train_batch_size=256, shuffle=True, **kwargs):
    ds = MNISTAugDataset
    ds_root = os.path.join(data_root, 'MNIST-aug500k-2')
    splits = [('train', train_batch_size, shuffle), ('valid', 256, False), ('test', 256, False)]
    return get_hdf5_loader(ds, ds_root, splits)

def get_mnist_aug_on(**kwargs):
    ds = MNISTAugDataset
    ds_root = os.path.join(data_root, 'MNIST-aug500k-2')
    splits = [('online', 1, False), ('valid', 256, False), ('test', 256, False)]
    return get_hdf5_loader(ds, ds_root, splits)

def get_mnist_aug_ds(**kwargs):
    ds = MNISTAugDataset
    ds_root = os.path.join(data_root, 'MNIST-aug500k-2')
    splits = [('shift', 1, False), ('valid', 256, False), ('test', 256, False)]
    return get_hdf5_loader(ds, ds_root, splits)

def get_mnist_aug_ds2(**kwargs):
    ds = MNISTAugDataset
    ds_root = os.path.join(data_root, 'MNIST-aug500k-2')
    splits = [('shift2', 1, False), ('valid', 256, False), ('test', 256, False)]
    return get_hdf5_loader(ds, ds_root, splits)

def get_mnist_aug_on_cut(sample, **kwargs):
    ds = MNISTAugDataset
    ds_root = os.path.join(data_root, 'MNIST-aug500k-2')
    splits = [('online', 1, False), ('valid', 256, False), ('test', 256, False)]
    return get_hdf5_loader(ds, ds_root, splits, sample=sample)

def get_mnist_aug_ds_cut(sample, **kwargs):
    ds = MNISTAugDataset
    ds_root = os.path.join(data_root, 'MNIST-aug500k-2')
    splits = [('shift', 1, False), ('valid', 256, False), ('test', 256, False)]
    return get_hdf5_loader(ds, ds_root, splits, sample=sample)


get = {
    'MNIST':get_mnist,
    #'MNISTAUG':get_mnist_aug500k,
    #'MNISTADS':get_mnist_aug500k_ds,
    #'MNISTADS50k':get_mnist_aug50k_ds,
    'MNISTATR':get_mnist_aug_off,
    'MNISTAON':get_mnist_aug_on,
    'MNISTADS':get_mnist_aug_ds,
    'MNISTADS100k':get_mnist_aug_ds2,
    'MNISTAON2k'  :lambda **kwargs: get_mnist_aug_on_cut(2000,   **kwargs),
    'MNISTAON10k' :lambda **kwargs: get_mnist_aug_on_cut(10000,  **kwargs),
    'MNISTAON100k':lambda **kwargs: get_mnist_aug_on_cut(100000, **kwargs),
    'MNISTADS100k':lambda **kwargs: get_mnist_aug_ds_cut(100000, **kwargs),
    'CIFAR10':get_cifar10,
    'SVHN':get_svhn,
}

