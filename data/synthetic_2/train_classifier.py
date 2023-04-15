""" Script to train classifier on synthetic data. """

import numpy as np
import pandas as pd
import torch
import wandb

from models.TCN import TCN_fit_pipeline, FastDataLoader
from models.TCN.utils import get_data_loader_synthetic

def get_random_hyperparameters():
    numpy_rng = np.random.default_rng()  # because the might be seed elsewhere in the code
    config = dict()
    config['num_layers'] = numpy_rng.choice([1,2,3], size=1, p=[0.5, 0.3, 0.2])
    config['channels_1'] = numpy_rng.choice([2,4,8,16,32,64,128], size=1)
    config['channels_2'] = numpy_rng.choice([2,4,8,16,32,64,128], size=1) if config['num_layers'] > 1 else 0
    config['channels_3'] = numpy_rng.choice([2,4,8,16,32,64,128], size=1) if config['num_layers'] > 2 else 0
    num_channels = [i[0] for i in [config['channels_1'], config['channels_2'], config['channels_3']] if i != 0]
    config['num_channels'] = num_channels
    config['dropout'] = torch.tensor(np.random.uniform(0, 0.5).__round__(2))
    return config


def hyper_search(runs: int):
    wandb.finish()
    path = "~/Documents/data/ts/synthetic_2/fully_observed/synthetic_2_ts_eav.csv.gz"
    batch_size = 2000
    drop_time = True
    input_features = 4
    dataloader, input_features, pos_weight = get_data_loader_synthetic(path, batch_size, drop_time)
    config = {
            'seed': 65,
            'model_type': 'TCN', 
            'lr': 0.001,
            'time excluded': (input_features==4),
            'num_inputs': input_features,
            'num_channels': [-1],
            'kernel_size': 2,
            'max_epochs': 1000,
            'batch_size': batch_size,
            'dropout': -1
            }
    
    for i in range(runs):
        print(f'------------- Run {i} out of {runs} ------------------')
        config.update(get_random_hyperparameters())
        print(config)
        TCN_fit_pipeline(dataloader, None, input_features, "classifier synthetic 2", pos_weight, config=config)
        
        
        
        

if __name__ == "__main__":
    hyper_search(10)
    