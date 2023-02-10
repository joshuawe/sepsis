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
from toy_dataset import data_utils

dataset = 'toy-josh'

batch_size_dict = {'physionet': 100, 'toy-josh': 512}
batch_size = batch_size_dict[dataset]

if dataset == 'toy':
    # data_obj = utils.get_synthetic_data(args)
    data_obj = utils.get_toydata(batch_size)
elif dataset == 'physionet':
    data_obj = utils.get_physionet_data(batch_size)
elif dataset == 'mimiciii':
    data_obj = utils.get_mimiciii_data(batch_size, filter_anomalies=True)
elif dataset == 'toy-josh':
    data_obj = utils.get_toydata(missingness_rate=0.2, batch_size=batch_size)
else:
    raise RuntimeError('error')

train_loader    = data_obj["train_dataloader"]
test_loader     = data_obj["test_dataloader"]
val_loader      = data_obj["val_dataloader"]
if dataset == 'toy-josh':
    gt_train_loader = data_obj["gt_train_dataloader"]
    gt_test_loader  = data_obj["gt_test_dataloader"]
    gt_val_loader   = data_obj["gt_val_dataloader"]
else:
    gt_train_loader = None
    gt_test_loader  = None
    gt_val_loader   = None

dim = data_obj["input_dim"]
union_tp = utils.union_time(train_loader)

def train_a_hetvae(model_args):
    try:
        hetvae = HETVAE(dim, union_tp, model_args)

        hetvae.train_model(train_loader, val_loader, gt_val_loader)

        ground_truth_batch, training_batch = data_utils.get_batch_train_ground_truth(train_loader, gt_train_loader, batch_num=0)
        pred_mean, preds, quantile_low, quantile_high = hetvae.impute(training_batch, 100, sample_tp=0.9)
        fig = utils.visualize_sample(training_batch, pred_mean, quantile_low, quantile_high, ground_truth_batch, sample=70)
        # hetvae.writer.add_image('Imputation', image)
        path = hetvae.log_path.joinpath('visualization.png')
        fig.savefig(path)

    except Exception as e:
        print(e)
        print('An Error occurred. \n\n\n\n\n')
        raise RuntimeError


""" ################################################################################################ """
""" ################################################################################################ """
""" ################################################################################################ """


# model_args = f'--niters 20 --lr 0.00005 --batch-size 128 --rec-hidden 128 --latent-dim 128 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --n 8000 --dataset {dataset} --seed 15 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp 0.3 --elbo-weight 1.0 --mse-weight 5.0 --mixing concat --k-iwae 10'
# train_a_hetvae(model_args)



for i in [0.1, 0.3, 0.9]:
    model_args = f'--niters 200 --lr 0.00005 --batch-size 128 --rec-hidden 128 --latent-dim 128 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --n 8000 --dataset {dataset} --seed 15 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp {i} --elbo-weight 1.0 --mse-weight 5.0 --mixing concat --k-iwae 10'
    train_a_hetvae(model_args)
    
# for i in np.arange(20,200, 10):
#     model_args = f'--niters 20 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 128 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --n 8000 --dataset {dataset} --seed 1 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight {i} --mixing concat --k-iwae 10'
#     train_a_hetvae(model_args)


# try:
#     model_args = f'--niters 20 --lr 0.0001 --batch-size 128 --rec-hidden 128 --latent-dim 128 --width 128 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --n 8000 --dataset physionet --seed 1 --save --norm --intensity --net hetvae --bound-variance --shuffle  --sample-tp 0.5 --elbo-weight 1.0 --mse-weight 5.0 --mixing concat --k-iwae 1'
#     hetvae = HETVAE(dim, union_tp, model_args)

#     hetvae.train_model(train_loader, val_loader, gt_val_loader)

#     ground_truth_batch, training_batch = data_utils.get_batch_train_ground_truth(train_loader, gt_train_loader, batch_num=0)
#     pred_mean, preds, quantile_low, quantile_high = hetvae.impute(training_batch, 100, sample_tp=0.9)
#     fig = utils.visualize_sample(training_batch, pred_mean, quantile_low, quantile_high, ground_truth_batch, sample=70)
#     # hetvae.writer.add_image('Imputation', image)
#     path = hetvae.log_path.joinpath('visualization.png')
#     fig.savefig(path)

# except Exception as e:
#     print(e)
#     print('An Error occurred. \n\n\n\n\n')
#     raise RuntimeError

