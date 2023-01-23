import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from random import SystemRandom
import models
import utils

from imputation.hetvae.src.train import HETVAE

dataset = 'toy-josh'
batch_size = 512

if dataset == 'toy':
    # data_obj = utils.get_synthetic_data(args)
    data_obj = utils.get_toydata(batch_size)
elif dataset == 'physionet':
    data_obj = utils.get_physionet_data(batch_size)
elif dataset == 'mimiciii':
    data_obj = utils.get_mimiciii_data(batch_size, filter_anomalies=True)
elif dataset == 'toy-josh':
    data_obj = utils.get_toydata(batch_size)
else:
    raise RuntimeError('error')

train_loader = data_obj["train_dataloader"]
test_loader = data_obj["test_dataloader"]
val_loader = data_obj["val_dataloader"]
dim = data_obj["input_dim"]
union_tp = utils.union_time(train_loader)

for mse_weight in [90]: # [10, 13 , 16, 19, 25, 30, 35, 45]:

    model_args = f'--niters 1500 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 128 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --n 8000 --dataset {dataset} --seed 1 --save --norm --intensity --net hetvae --bound-variance  --sample-tp 0.85 --elbo-weight 1.0 --mse-weight {mse_weight} --mixing concat --k-iwae 10'
    hetvae = HETVAE(dim, union_tp, model_args)

    hetvae.train_model(train_loader, val_loader)
