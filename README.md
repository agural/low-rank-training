# Low Rank Training of Deep Neural Networks for Emerging Memory Technology

This directory contains files associated with low rank training research. There are two primary development paths: a PyTorch version for fast batch training to test high-level ideas in `pytorch` and a NumPy version for implementing low rank training in `lr`. Runs of either path produce debug output files in `analysis` and scripts in that folder allow for interesting analysis of training behavior.

## Dependencies
The main dependencies are shown below.
1. `Python 3.6`
1. `PyTorch 1.1.0`
1. `TorchVision 0.2.2`
1. `numpy`, `scipy`, `matplotlib`, `h5py`
1. `numba`, `profilehooks`, `opencv-python`
    1. Note that `numba` may require additional installs such as llvm.

## Main Contents
1. `lr/main.py`: The main LR training file (see file for argument options).
1. `pytorch/main.py`: Script for running a PyTorch version to get initial weights (see file for argument options).
1. `analysis/Plots.ipynb`: Experiment plots used in the paper.

## Setup and Run
### Initial Setup and Baseline Experiment Run
1. `cd` to the root directory `LR_train` and activate the python environment (if any).
1. `$ pip install -r requirements.txt`
1. `$ pip install -e .`
1. `$ ./run.sh lrt-base`

### Generate the Dataset
1. `$ cd data`
1. `$ jupyter notebook .`
1. Open `MNIST Dataset Generator.ipynb`
1. Run all cells (should take less than 2 hours).

### Run Low Rank Training on the Augmented Dataset
Find the ID of the experiment you would like to run in `./lr/experiments.py` which should have the form `lrt-xxx*`.

1. `cd` to the root directory `LR_train`.
1. `./run.sh lrt-xxx*`

### Generate the PyTorch initialization model
1. `cd` to the root directory `LR_train`
1. `./pytorch/experiments.sh`

## Credits
[Redacted](https://github.com/redacted/low-rank-train)

