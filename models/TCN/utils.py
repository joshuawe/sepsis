""" UTIL Functions for the TCN model """	

import pandas as pd
import numpy as np
import torch

from .fast_data_loader import FastDataLoader


def get_data_loader_synthetic(
            path: str, 
            batch_size: int, 
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
    drop_time : bool, optional
        Should time be exluded from the input features, by default True

    Returns
    -------
    tuple[FastDataLoader, int, torch.Tensor]
        Dataloader, number of input features, class imbalance factor
    """
    
    df = pd.read_csv(path, compression="gzip")
    df = pd.pivot_table(df, index=["id", "time"], columns="variable", values="value").reset_index(level=[0, 1])
    if drop_time == True:
        df = df.drop(columns=["time"])
    print(df.shape)
    grouped = df.groupby("id")
    input_features = len(df.columns) - 2
    print(df.columns, '\n')
    print(df.tail(), '\n')
    print('Number of samples: ', len(grouped))

    # Get batches
    batch_representation = np.stack([i[1].iloc[:,1:] for n, i in enumerate(grouped)])
    batch_representation = batch_representation.astype(np.float64)

    # get dataloader
    dataloader = FastDataLoader(batch_representation, batch_size, 1)
    print("Batch.shape: ", next(iter(dataloader)).shape)
    input_features = next(iter(dataloader)).shape[-1] -1  # -1 for the label
    
    # get class imbalance factor
    prop_cases = df['Y_ts'].mean()
    pos_weight = torch.tensor((1 - prop_cases) / prop_cases)
    print('pos_weight: ', pos_weight)
    return dataloader, input_features, pos_weight



def get_dataloader_miiv(
            path: str, 
            batch_size: int, 
            drop_time: bool=True
            ) -> "tuple[FastDataLoader, int, torch.Tensor]":
    raise NotImplementedError




def fix_missingness_in_data():
    raise NotImplementedError