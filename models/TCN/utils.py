""" UTIL Functions for the TCN model """	

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.stats import zscore

import gc
from typing import Optional, Tuple, Union

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



def get_data_loader_miiv(
            path: Optional[str]=None, 
            batch_size: int=100,
            splits: tuple= (0.8, 0.1, 0.1),
            drop_time: bool=True,
            pad_value=-1
            ):
    
    if path is None:
        path = '/home2/joshua.wendland/Documents/sepsis/notebooks/miiv/miiv_fully_observed_TCN.npz'
        
    file_path = Path(path)
    if file_path.is_file():
        print('Loading data from file ...')
        batch_representation = np.load(file_path, allow_pickle=True)['data']
    else:
        raise FileNotFoundError(f'File not found: {file_path}')
    

    print('Batches shape:', batch_representation.shape)
    
    num_samples = batch_representation.shape[0]
    train_num = int(splits[0] * num_samples)
    val_num = int((splits[0] + splits[1]) * num_samples)
    dataloader_train = FastDataLoader(batch_representation[:train_num], batch_size, 1)
    dataloader_val = FastDataLoader(batch_representation[train_num:val_num], batch_size, 1)
    dataloader_test = FastDataLoader(batch_representation[val_num:], batch_size, 1)
    
    input_features = batch_representation.shape[-1] -1  # -1 for the label
    
    prop_cases = batch_representation[:, :, -1].mean(where=(batch_representation[:, :, -1] != pad_value))
    pos_weight = torch.tensor((1 - prop_cases) / prop_cases)
    
    return dataloader_train, dataloader_val, dataloader_test, input_features, pos_weight




def fix_missingness_in_data():
    raise NotImplementedError



def preprocess_data_for_TCN(path_read: str, path_save: str, pad_value: int=-1, fill_na_value=0, zscore_flag=True) -> dict:
    """
    Preprocesses the mimiv 4 data for the TCN model. It expects a parquet file with the data in wide format. It will fill missing values with (default: ffill, then 0), rearrange the label column (put it in last position) and drop the time column. It will then save the data in a npz file with the key 'data'. If zscores is True, it will zscore the data and save the mean and std as in the npz with the respective keys.  
    Eventhough the function returns the data dict, it is recommended to first save the data and then load it again, as there might be some data leakage otherwise, consuming a lot of RAM memory.

    Parameters
    ----------
    path_read : str
        Path to read the data from. Is expected to be in wide format.
    path_save : str
        Patht to save the data to.
    pad_value : int, optional
        Value used to pad all sequences to the same length, by default -1
    fill_na_value : int, optional
        All values that remain as NAN after applying ffill will be filled with `fill_na_value`, by default 0
    zscore_flag : bool, optional
        Wether to compute the zscore *and* to perform zscore on data, by default True
        
    Returns
    ---------
    dict:
        Dictionary with the keys 'mean' and 'std' if zscore_flag is True, and 'data'.
    """
    # miiv_path_p = '~/Documents/data/ts/miiv/fully_observed/miiv_ts_wide.parquet'
    # save_path = '/home2/joshua.wendland/Documents/sepsis/notebooks/miiv/miiv_fully_observed_TCN_zscore.npz'
    df = pd.read_parquet(path_read)
    
    # rearrange label column and drop time
    cols = list(df.columns)
    cols.remove('label')
    cols.remove('time')
    cols += ['label']
    df = df[cols]
    print('Rearranged label column and dropped time column.')
    
    # zscore data
    if zscore_flag is True:
        # select only the numeric columns
        numeric_cols = list(df.select_dtypes(include=np.number).columns)
        numeric_cols.remove('label')  # get rid of label, we do not want to zscore it
        numeric_cols.remove('id')  # also no zscore for IDs (which can be numeric)
        mean = df[numeric_cols].mean()
        std = df[numeric_cols].std(ddof=0)
        df[numeric_cols] = (df[numeric_cols] - mean) / std
        print('Zscored data.')
        
    # fill missing values
    df = df.groupby('id', group_keys=False).apply(lambda x: x.ffill().fillna(value=fill_na_value))
    print(f'Filled missing values (ffill, then fillna (value={fill_na_value})).')
    
    # pad sequences
    batch_representation = pad_sequence(df, 170, pad_value=pad_value)
    
    # save data
    data_dict = {'data': batch_representation}
    if zscore_flag is True:
        data_dict['mean'] = mean
        data_dict['std'] = std
    np.savez(path_save, **data_dict)
    print('batch_representation.shape:', batch_representation.shape)
    print('Saved numpy object.')
    return data_dict


def pad_sequence(x: pd.DataFrame, max_seq_len: int, pad_value = np.nan) -> np.ndarray:
    """
    Returns 3D numpy array (`num_sequences x num_timepts x num_features`) for predictor model training andtesting after padding. Input is a pandas DataFrame in wide format with columns "id" and "time" andother features. The sequences in the DataFrame are grouped by id and are padded to the same length. If`self.max_seq_len`is shorter than a sequence length, the sequence is clipped to `self.max_seq_len`.Then the sequences are stacked into a 3D numpy array which is returned. During this process the "id"column is dropped.

    Parameters
    ----------
    x : pd.DataFrame
        Input data. Must have columns "id" and "time" and other features. Wide format.  

    Returns
    -------
    np.ndarray
        The data grouped by id, padded to the same length, and stacked into a 3D numpy array. Shape is (`num_sequences x num_timepts x num_features`).
    """
    # group by id
    grouped = x.groupby('id')
    
    # Initialize a list to store the padded sequences
    padded_sequences = []

    # Loop through the groups, pad each sequence, and append to the list
    for i, (_, group) in tqdm(enumerate(grouped), desc='Padding sequences', total=len(grouped)):
        # Calculate the number of rows to pad with zeros
        num_padding_rows = max_seq_len - len(group)
        
        # if padding is necessary
        if num_padding_rows > 0:    
            # Create a DataFrame of zeros with the same column structure as the group
            padding_rows = np.empty((num_padding_rows, group.shape[1]))
            padding_rows[:] = pad_value
            padding_rows = pd.DataFrame(padding_rows,
                                        columns=group.columns)
            
            # Concatenate the group and the padding_rows DataFrames
            padded_group = pd.concat([group.copy(), padding_rows], ignore_index=True)
            
            # Convert the padded group to numpy array and remove the 'id' column
            padded_numpy = padded_group.drop(columns=['id']).to_numpy()
            
        # if padding is not necessary, maybe clipping is
        else:
            padded_numpy = group.iloc[:self.max_seq_len]
            padded_numpy = padded_numpy.drop(columns=['id']).to_numpy()
        
        # Append the numpy array to the list
        padded_sequences.append(padded_numpy)
        
        # Garbage collection after every 10% of the groups
        if (i % int(0.09 * len(grouped)) == 0):
            gc.collect()

    # Stack the sequences to obtain the final numpy tensor
    stacked = np.stack(padded_sequences, axis=0)
    
    return stacked.astype(np.float32)