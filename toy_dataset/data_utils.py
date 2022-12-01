"""Utility functions for Toydataset / Synthetic dataset.
    Multivarite time series.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pycorruptor as pc

class ToyDataset(Dataset):
    """Dataset class for toy / synthetical data to test before continuing with EHR data. It is meant for multivariate time series.

            + `self.x_data`: Is a Pandas Dataframe, holding a list of time series, which also are Dataframes.
            + `self.x_ids`: Is a Pandas Dataframe, holding the corresponding id to each time series of `self.x_data`.

    Args:
        Dataset: The base Pytorch Dataset class.
    """

    def __init__(self, path, missingness=None, missingness_rate=0.3, missingness_value=-1):
        """Read in the data from *.csv file and prepare it. After init the class is left with self.x_data and self.x_ids.

        Args:
            path (str): Path to csv file.
            missingness (str): The missingness pattern. Possible options are mcar and None. Defaults to None. If the pycorruptor module implements the mar and mnar cases, they can be added as well. 
            missingness_rate (float): If `missigness!=0`, then this value decides the amount of missing data in percentage, should be [0,1].
            missingness_value (float): The replacement value for all data points, where missingness should be induced.
        """
        # set missingness function
        if missingness == 'mcar':
            self.missingness = pc.mcar
        else:
            self.missingness = None
        self.missingness_rate = missingness_rate
        self.missingness_value = missingness_value
        # Read in csv
        df = pd.read_csv(path, compression=None)
        # Sort wrt id, then sort wrt time, ascending
        df = df.sort_values(by=['id', 'time'], ascending=True, ignore_index=True)
        # Number of samples
        self.n_samples = len(df['id'].unique())
        # Group by ID
        df_grouped = pd.DataFrame(df.groupby('id'))
        # Get rid of the 'id'-column in each group
        for x in df_grouped.iloc[:,1]:
            x.drop(columns=['id'], inplace=True)
        # extract the time series
        self.x_data = df_grouped.drop(columns=0)
        self.x_ids = df_grouped.drop(columns=1)

    def __getitem__(self, index):
        """Returns items from dataset for given index. Due to the nested structure of pd.DataFrames and np.arrays, some overhead instructions are necessary.

        Args:
            index (): Index for data sample(s).

        Returns:
            tuple: sample of data, id.
        """
        # Get the data from indexes
        X = self.x_data.loc[index]
        x_ids = self.x_ids.loc[index]
        x_ids = x_ids.tolist()
        # Extract, if multiple data, ...
        if isinstance(index, slice):
            x = list()
            for time_series in X[1]:
                # ... convert to numpy and append to list ...
                x.append(time_series.to_numpy())
            # ... and create one big Tensor array
            x = np.array(x) # first converting to numpy is much faster
            x = torch.tensor(x)
        else:
            x = X.loc[1]
            x = torch.tensor(x.values)
            # x = x[None, :, :]  # same as unsequeeze

        # add the missingness if necessary
        if self.missingness is not None:
            # X_intact = Original input
            # X = Original input with artificial missingness
            # missing_mask = Mask indicating all missing values in X
            # indicating_mask = Mask indicating all artificially missing values in X
            X_intact, X, missing_mask, indicating_mask = self.missingness(x, 
                                                                            self.missingness_rate, 
                                                                            self.missingness_value)
            Y = X_intact
            
        
        sample = X, x_ids, Y
        return sample


    def __len__(self):
        """The number of samples (time series) available in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.n_samples

    