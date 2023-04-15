""" UTIL Functions for the TCN model """	

import pandas as pd
import numpy as np
import torch

from .fast_data_loader import FastDataLoader


def get_data_loader_synthetic(
            path: str, 
            batch_size: int,
            splits: list,
            drop_time: bool=True
            ) -> "tuple[FastDataLoader, int, torch.Tensor]":
    """
    Get the dataloader for the synthetic data. Also gives the number of input features and the class imbalance factor.

    Parameters
    ----------
    path : str
        Path to the dataset. Is expected to be in EAV format, not wide.
    batch_size : int
        Number of samples per batch.
    splits : list
        List of splits to use. Should be a list of 3 floats, summing to 1. Train, val, test. Note: Only the first two are used and the remaining is used for test.
    drop_time : bool, optional
        Should time be exluded from the input features, by default True

    Returns
    -------
    tuple[FastDataLoader, int, torch.Tensor]
        Dataloader, number of input features, class imbalance factor
    """
    
    df = pd.read_csv(path, compression="gzip")
    df = pd.pivot_table(df, index=["id", "time"], columns="variable", values="value").reset_index(level=[0, 1])
    
    # impute and fill all missing values with 0
    df = df.groupby('id', group_keys=False).apply(lambda x: x.ffill().fillna(value=0))
    if drop_time == True:
        df = df.drop(columns=["time"])
    print(df.shape)
    grouped = df.groupby("id")
    input_features = len(df.columns) - 2
    num_samples = len(grouped)
    print(df.columns, '\n')
    print(df.tail(), '\n')
    print('Number of samples: ', num_samples)

    # Get batches
    batch_representation = np.stack([i[1].iloc[:,1:] for n, i in enumerate(grouped)])
    batch_representation = batch_representation.astype(np.float64)

    # get dataloader
    train_num = int(splits[0] * num_samples)
    dataloader_train = FastDataLoader(batch_representation[:train_num], batch_size, 1)
    dataloader_val = FastDataLoader(batch_representation[train_num:], batch_size, 1)
    print('Careful, only using the first two splits for train and val, no test split is used!')
    print("Batch.shape: ", next(iter(dataloader_train)).shape)
    input_features = next(iter(dataloader_train)).shape[-1] -1  # -1 for the label
    
    # get class imbalance factor
    prop_cases = df['Y_ts'].mean()
    pos_weight = torch.tensor((1 - prop_cases) / prop_cases)
    print('pos_weight: ', pos_weight)
    return dataloader_train, dataloader_val, input_features, pos_weight



def get_dataloader_miiv(
            path: str, 
            batch_size: int, 
            drop_time: bool=True
            ) -> "tuple[FastDataLoader, int, torch.Tensor]":
    raise NotImplementedError




def fix_missingness_in_data():
    raise NotImplementedError