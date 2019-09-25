#!/bin/bash

# These are the commands run to generate the PyTorch models (run from root dir).
python pytorch/main.py --epochs=30 --batch_size=256 --lr=0.0025 --opt='Adam' --quantize_after=25 --save_model='MNISTATR_model.pt' --dataset='MNISTATR' --name='pt-004' --description='PT/Minibatch/Adam model trained on MNISTATR.'

