"""Utility functions for Toydataset / Synthetic dataset.
    Multivarite time series.
"""
import os 
import sys
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pycorruptor as pc
# from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt

from pypots.imputation import SAITS, BRITS
from sklearn.preprocessing import StandardScaler
from pypots.data import mcar, masked_fill
from pypots.utils.metrics import cal_mae, cal_mse

datasets_dict = {'toydataset_small': {
                                'descr': 'A very small and simple dataset, that is used for initial testing, not even a proper toy data set, very small.',
                                'path': '/home2/joshua.wendland/Documents/sepsis/toy_dataset/synthetic_ts_1/synthetic_ts_test_data_eav.csv.gz'
                                },
                'toydataset_50000': {
                                'name': 'Synthetic Time Series (4 Vars, 50000 samples)',
                                'descr': 'A simple synthetical time series data set with 50000 data samples. Columns include id, time, noise, trend, seasonal, trend+season.',
                                'path_train': '/home2/joshua.wendland/Documents/sepsis/toy_dataset/synthetic_ts_4types_50000/synthetic_ts_train_40000.csv.gz',
                                'path_validation': '/home2/joshua.wendland/Documents/sepsis/toy_dataset/synthetic_ts_4types_50000/synthetic_ts_validation_5000.csv.gz',
                                'path_test': '/home2/joshua.wendland/Documents/sepsis/toy_dataset/synthetic_ts_4types_50000/synthetic_ts_test_5000.csv.gz'
                                }
}

class ToyDataset(Dataset):
    """Dataset class for toy / synthetical data to test before continuing with EHR data. It is meant for multivariate time series.

            + `self.x_data`: Is a Pandas Dataframe, holding a list of time series, which also are Dataframes.
            + `self.x_ids`: Is a Pandas Dataframe, holding the corresponding id to each time series of `self.x_data`.

    Args:
        Dataset: The base Pytorch Dataset class.
    """

    def __init__(self, path, missingness='mcar', missingness_rate=0.3, missingness_value=-1):
        """Read in the data from *.csv file and prepare it. After init the class is left with self.x_data and self.x_ids.

        Args:
            path (str): Path to csv file.
            missingness (str): The missingness pattern. Possible options are mcar and None. Defaults to None. If the pycorruptor module implements the mar and mnar cases, they can be added as well. 
            missingness_rate (float): If `missigness!=0`, then this value decides the amount of missing data in percentage, should be [0,1].
            missingness_value (float): The replacement value for all data points, where missingness should be induced.
        """
        # set missingness function
        if missingness == 'mcar':
            pass
        else:
            self.missingness_rate = 0
        
    
        self.missingness = pc.mcar
        self.missingness_rate = missingness_rate
        self.missingness_value = missingness_value
        if self.missingness_rate in [0,1]: print(f"Attention, missingness_rate = {self.missingness_rate} !")
        # Read in csv
        df = pd.read_csv(path, compression=None)
        self.input_dim = len(df.columns) - 2  # input dimenstion without 'id' and 'time' column.
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
        
        The returned X is of dimensions [B x T x N].  
        + B = Batch size
        + T = Num time points
        + N = 2 * features + 1 (observed data (features) + mask of observed data (features) + observed time points (1D)). The observed time points contain the actual time corresponding to the datapoint.

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
            raise NotImplementedError('Did not implement the time extraction thing for slices in custom DataLoader.')
            x = list()
            for time_series in X[1]:
                # ... convert to numpy and append to list ...
                x.append(time_series.to_numpy())
            # ... and create one big Tensor array
            x = np.array(x) # first converting to numpy is much faster
            x = torch.tensor(x)
        else:
            x = X.loc[1]
            time = x['time']
            time = torch.tensor(x['time'].values)
            x = x.drop(columns=['time'])
            x = torch.tensor(x.values)
            # x = x[None, :, :]  # same as unsequeeze

        # add the missingness (even if no missingness required)
        # X_intact = Original input
        # X = Original input with artificial missingness
        # missing_mask = Mask indicating all missing values in X
        # indicating_mask = Mask indicating all artificially missing values in X
        X_intact, X, missing_mask, indicating_mask = self.missingness(
                                                                    x, 
                                                                    self.missingness_rate, 
                                                                    self.missingness_value
        )
        # X_intact = X_intact * missing_mask

        # concatenate all information into X
        # time = time.unsqueeze(1)
        # X_concat = torch.concatenate((X, indicating_mask, time), dim=1)
        
        # sample = X_concat, x_ids, Y
        # X = torch.tensor(X.clone().detach(), dtype=torch.float32)
        X = X.type(torch.float32)
        missing_mask = missing_mask.type(torch.float32)
        time = time.type(torch.float32)
        # sample = X, missing_mask, time, Y
        sample = torch.concatenate((X, missing_mask, time.unsqueeze(1)), dim=1)
        sample = sample.type(torch.float32)
        return sample


    def __len__(self):
        """The number of samples (time series) available in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.n_samples

    
def get_Toy_Dataloader(path, missingness=None, missingness_rate=0.3, missingness_value=-1, batch_size=1, shuffle=False, **kwargs) -> 'tuple[DataLoader, DataLoader]':
        """Wrapper function for the Toydataset in order to get the DataLoader directly.
        Returns the expected `dataloader` together with a `ground_truth_dataloader`, which does not contain any missingness. This is of course only the case, when the `missingness_rate != 0`. If it is 0, then the `ground_truth_loader` is `None`.

        Args:
            path (str): Path to dataset.
            missingness (str, optional): Check out ToyDataset class. Defaults to None.
            missingness_rate (float, optional): Defaults to 0.3.
            missingness_value (float, optional): Defaults to -1.
            batch_size (int, optional):  Defaults to 1.
            shuffle (bool, optional): Shuffle the samples in DataLoader. Defaults to False.

        Returns:
            tuple(torch.DataLoader, torch.DataLoader): The Dataloaders for the Toy Dataset. Regular dataloader together with ground_truth_loader.
        """
        # create dataset
        dataset = ToyDataset(path, missingness=missingness, missingness_rate=missingness_rate, missingness_value=missingness_value)

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, prefetch_factor=12, num_workers=3)
        
        if missingness_rate != 0:
            ground_truth_dataset = ToyDataset(path, missingness=None, missingness_rate=0, missingness_value=missingness_value)
            ground_truth_loader = DataLoader(dataset=ground_truth_dataset, batch_size=batch_size, shuffle=shuffle, prefetch_factor=12, num_workers=3)
        else:
            ground_truth_loader = None

        return dataloader, ground_truth_loader

def get_Toydata_df(path):
    """Fetches the Toy Dataset from `path` and returns a `pandas.DataFrame`.

    Args:
        path (str): Path to dataset.

    Returns:
        pandas.DataFrame: The Dataset.
    """
    df = pd.read_csv(path, compression=None)

    return df


class ToyDataDf():
    def __init__(self, path=None, name='Toydataset'):
        """Fetches the Toy Dataset from `path` and returns a `pandas.DataFrame`.

        Args:
            path (str, dict): Path to dataset. Or dict containing all information. See `self.set_up_with_dict()`.
        """
        self.path = path
        if isinstance(path, str):
            self.df = pd.read_csv(path, compression=None)
            self.df = self.df.sort_values(by=['id', 'time'], ascending=True, ignore_index=True)  # time was not sorted
            self.n_features  = len(self.df.columns) - 2 
            self.ids = self.df['id'].unique()
            self.num_samples = len(self.ids)
        elif isinstance(path, dict):
            self.set_up_with_dict(path)
        self.name = name
        self.artificial_missingness = None
        self.artificial_missingness_rate = None
        return

    def set_up_with_dict(self, path:dict):
        self.path_train = path['path_train']
        self.path_validation = path['path_validation']
        self.path_test = path['path_test']
        self.name = path['name']
        self.df = pd.read_csv(self.path_train, compression=None)
        self.df = self.df.sort_values(by=['id', 'time'], ascending=True, ignore_index=True)
        self.n_features  = len(self.df.columns) - 2 
        self.ids = self.df['id'].unique()
        self.num_samples = len(self.ids)


    def __len__(self) -> int:
        return self.num_samples

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.__str__()

    def get_sample(self, index:int):
        """Get a single sample of dataset corresponding to the index.

        Args:
            index (int): Sample index.

        Returns:
            X_intact (np.array):  Ground truth. (time x features)
            X (np.array): Data with artificially induced missingness. (time x features)
            ind_mask (np.array): Indicating mask, =1 where missing data, =0 available data. (time x features)
            time_pts (np.array): The time points for the data. (time x 1)
        """
        df = self.df
        id = self.ids[index]
        # X_intact and time_pts
        X_intact = df.loc[df['id'] == id]
        time_pts = X_intact.iloc[:,1].to_numpy(copy=True)
        X_intact = X_intact.iloc[:,2:].to_numpy(copy=True)
        # X (with artificial missingness)
        X = self.df_mis[df['id'] == id]
        X = X.iloc[:,2:].to_numpy(copy=True)
        # ind_mask 
        ind_mask = self.ind_mask[df['id']==id]
        ind_mask = ind_mask[:,2:]
        return X_intact, X, ind_mask, time_pts, id

    def create_mcar_missingness(self, missingness_rate, missingness_value=np.nan, verbose=True) -> None:
        """Creates MCAR missingness on `self.df`. The first two columns are not included in the missingness process, as they are assumed to be 'id' and 'time'. The dataset with missingness and the corresponding mask are saved in `self.df_mis` and `self.ind_mask`. The shape of `self.df`, `self.df_mis` and `self.ind_mask` is the same.

        Args:
            missingness_rate (float): The amount of additionally missing data. Between 0.0 and 1.0 .
            missingness_value (float, optional): The value the missing datapoints are to be assigned. Defaults to np.nan.
            verbose (bool, optional): Print information to console. Defaults to False.
        """
        assert (0 <= missingness_rate < 1), f'Missing rate has to be in interval [0,1), instead got {missingness_rate}'
        df = self.df.copy()
        # create missingness in data
        df_intact, df_mis, miss_mask, ind_mask = pc.mcar(df.iloc[:,2:].to_numpy(), missingness_rate, missingness_value)
        # ind_mask = (ind_mask==0) * 1 # this inverts the ind_mask
        num_values = miss_mask.size
        # add the missingdata back into df
        df.iloc[:,2:] = pd.DataFrame(np.array(df_mis), columns=df.columns[2:])
        # Add the two discarded columns to the miss_mask and ind_mask
        ones = np.ones((miss_mask.shape[0],2))
        miss_mask = np.concatenate((ones, miss_mask), axis=1)
        ind_mask = np.concatenate((ones, ind_mask), axis=1)
        # save data to class
        self.df_mis = df
        self.ind_mask = ind_mask
        self.artificial_missingness = 'mcar'
        self.artificial_missingness_rate = missingness_rate
        self.artificial_missingness_value = missingness_value
        if verbose:
            print(f'--\nCreated MCAR missing data, but without missingness in columns {df.columns[:2]}')
            print(f'missingness_rate: {missingness_rate},\tmissingness_value: {missingness_value}')
            num_mis = df.isna().sum().sum()
            print(f'Missing values: {num_mis} out of {num_values} ({num_mis/num_values:.1%}) (!excluding aforementioned columns)')
            print(f'Data values in entire dataframe is {df.size} (shape: {df.shape})')
        return

    def get_missingness_data(self, for_mtan=False):
        """Receive the missingness generated for this dataset in the form of `X_intact, X, indicating_mask`. Beware of the datatypes. No exluded columns or so (= including id and time).

        Args:
            for_mtan (bool): Flag. Should data be provided for mtan format

        Returns:
            tuple(pd.Dataframe, pd.Dataframe, np.ndarray): X_intact, X, indicating_mask
        """
        X_intact = self.df
        X = self.df_mis
        indicating_mask = self.ind_mask
        data = X_intact, X, indicating_mask
        return data

    def impute_mean(self, X:pd.DataFrame=None):
        """Perform mean imputation on the dataset. Only works if missingness has been created already.

        Args:
            X (pd.DataFrame, optional): Data containing missingness. Missing values should be NANs.

        Returns:
            pd.DataFrame: The dataset with mean imputation.
        """
        imputation_func = pd.DataFrame.fillna
        imputation_func = 'mean'
        # helper_func = pd.DataFrame.mean
        kwargs = {'numeric_only':True}
        return self._base_impute(imputation_func, X, **kwargs)

    def impute_median(self, X:pd.DataFrame=None):
        """Perform median imputation on the dataset. Only works if missingness has been created already.

        Args:
            X (pd.DataFrame, optional): Data containing missingness. Missing values should be NANs.

        Returns:
            pd.DataFrame: The dataset with median imputation.
        """
        imputation_func = pd.DataFrame.fillna
        imputation_func = 'median'
        # helper_func = pd.DataFrame.median
        kwargs = {'numeric_only':True}
        return self._base_impute(imputation_func, X, **kwargs)

    def impute_LOCF(self, X:pd.DataFrame=None):
        """Perform LOCF (last observation carried forward) imputation on the dataset. Only works if missingness has been created already.

        Args:
            X (pd.DataFrame, optional): Data containing missingness. Missing values should be NANs.

        Returns:
            pd.DataFrame: The dataset with LOCF imputation.
        """
        imputation_func = pd.DataFrame.fillna
        imputation_func = 'fillna'
        kwargs = {'method':'ffill'}
        return self._base_impute(imputation_func, X, **kwargs)

    def impute_NOCB(self, X:pd.DataFrame=None):
        """Perform NOCB (next observation carried backwards) imputation on the dataset. Only works if missingness has been created already.

        Args:
            X (pd.DataFrame, optional): Data containing missingness. Missing values should be NANs.

        Returns:
            pd.DataFrame: The dataset with NOCB imputation.
        """
        imputation_func = pd.DataFrame.fillna
        imputation_func = 'fillna'
        kwargs = {'method':'backfill'}
        return self._base_impute(imputation_func, X, **kwargs)

    def _base_impute(self, imputation_func, X:pd.DataFrame=None, helper_func=None, **kwargs):
        """Base function for simple imputation methods. Groups the time series in the dataset by 'id' and the performs the corresponding imputation method that was passed as argument in `imputation_func`.

        Args:
            imputation_func (str): The string of *pandas.DataFrame* function that performes desired imputation method.
            X (pd.DataFrame, optional): External missing data, for imputation.
            helper_func (class function): A helper function for the imputation to be executed.
            kwargs: The kwargs that are passes as arguments to the `imputation_func`.

        Raises:
            RuntimeError: Only, if no missingness has been created already.

        Returns:
            pd.DataFrame: Dataframe with the imputed values.
        """
        if self.artificial_missingness is None:
            raise RuntimeError('First create missingness.')
        # Get helper_function arguments if they exist
        # kw = kwargs.pop('helper', None)
        # Check if external data should be used for imputation
        if X is None:
            df = self.df_mis.copy() 
        else:
            df = X.copy()

        df = df.fillna(df.groupby('id').transform(imputation_func, **kwargs))


        # # get all time series IDs
        # IDs = df['id'].unique()
        # # Cycle through each time series ID
        # for id in tqdm(IDs, desc='Imputing ID'):
        #     # get only the time series with corresponding ID
        #     ts = df.loc[df['id'] == id]
        #     # execute helper function if necessary
        #     if helper_func is not None:
        #         value = helper_func(ts, **kw)
        #     else:
        #         value = None
        #     # get the imputation value for each column
        #     impu = imputation_func(ts, value=value, **kwargs)
        #     imputed = X.fillna(X.groupby('id').transform('fillna', method='ffill'))
        #     # replace nan values with imputed values
        #     df.loc[df['id'] == id] = impu # ts.fillna(impu)
        return df

    def _base_impute_helper(self):
        pass

    def mse(self, imputed_df:pd.DataFrame, percent=False):
        """Calculates the MSE (mean squared error) between the imputed data in `imputed_df` and the original data in `self.df`.

        Args:
            imputed_df (pd.DataFrame): Dataset with imputed values.

        Raises:
            RuntimeError: Only, if no missingness has been created already.

        Returns:
            float: MSE
        """
        if self.artificial_missingness is None:
            raise RuntimeError('First create missingness.')
        # get rid of columns: id, time
        X = self.df.iloc[:, 2:]
        Y = imputed_df.iloc[:, 2:]
        assert(X.shape == Y.shape)
        # calculate mse
        mse = (X - Y)**2
        if percent is True:
            mse = mse / X
        # sum = mse.sum().sum()
        # only consider values that had been missing before
        #    Not necessary here, but good practice, as later on it will be necessary.
        mse *= self.ind_mask[:, 2:]
        mse =  mse.sum().sum()
        mse /= self.ind_mask[:, 2:].sum().sum()

        return mse
    
    
    def mae(self, imputed_df:pd.DataFrame, percent=False):
        """Calculates the MAE (mean absolute error) between the imputed data in `imputed_df` and the original data in `self.df`.

        Args:
            imputed_df (pd.DataFrame): Dataset with imputed values.

        Raises:
            RuntimeError: Only, if no missingness has been created already.

        Returns:
            float: MAE
        """
        if self.artificial_missingness is None:
            raise RuntimeError('First create missingness.')
        # get rid of columns: id, time
        X = self.df.iloc[:, 2:]
        Y = imputed_df.iloc[:, 2:]
        assert(X.shape == Y.shape)
        # calculate mse
        mae = (X - Y).abs()
        if percent is True:
            mae = mae / X
        # sum = mae.sum().sum()
        # only consider values that had been missing before
        #    Not necessary here, but good practice, as later on it will be necessary.
        mae *= self.ind_mask[:, 2:]
        mae =  mae.sum().sum()
        mae /= self.ind_mask[:, 2:].sum().sum()

        return mae
    
    def bias(self, imputed_df:pd.DataFrame, percent=False):
        if self.artificial_missingness is None:
            raise RuntimeError('First create missingness.')
        # get rid of columns: id, time  
        X = self.df.iloc[:, 2:]
        Y = imputed_df.iloc[:, 2:]
        assert(X.shape == Y.shape)
        # calculate mse
        bias = (Y - X)
        if percent is True:
            bias = bias / X
        # sum = mae.sum().sum()
        # only consider values that had been missing before
        #    Not necessary here, but good practice, as later on it will be necessary.
        bias *= self.ind_mask[:, 2:]
        bias =  bias.sum().sum()
        bias /= self.ind_mask[:, 2:].sum().sum()
        return bias
    
    @staticmethod
    def calc_bias(X, Y, mask, keep_vars=True):
        """Returns raw bias and percent bias between X and Y.

        Args:
            X (any): Original values X.
            Y (any): New or predicted values of X, Y.
            mask (any): An array indicating, at which locations should be computed. (1=compute, 0=ignore).
            keep_vars (bool): Whether the average over all variables should be calculated or not. Defaults to False.

        Returns:
            any, any: raw_bias and percent_bias
        """
        assert(type(X) == type(Y)), f'Input of X, Y is not of same type, instead {type(X)}, {type(Y)} and mask is {type(mask)}'
        X, Y, mask = np.array(X), np.array(Y), np.array(mask)
        raw_bias     = (Y - X)
        percent_bias = raw_bias / (X + (X==0))   # avoid devision by zero
        raw_bias     *= mask
        percent_bias *= mask * (X != 0)
        if keep_vars is False:
            axis = None
        else:
            axis = tuple(range(raw_bias.ndim-1))
        no_nans = ~np.isnan(percent_bias)
        percent_bias = percent_bias.sum(axis=axis, where=no_nans) / mask.sum(axis=axis, where=no_nans)
        no_nans = ~np.isnan(raw_bias)
        raw_bias     = raw_bias.sum(axis=axis) / mask.sum(axis=axis, where=no_nans)
        return raw_bias, percent_bias
        
    

    def get_mse_mae_impute(self, name, error_dict, impute_func, missingness_rates, repeat_imputation=1):
        """Calculates the (average) MSE and MAE for a given imputation method passes as the argument `impute_func`.

        Args:
            name (str): Name of imputation method.
            error_dict (dict): A dict into which the new errors can be entered via `error_dict[name] = mse_imputation_method`.
            impute_func (class function): The imputation function to be executed. It should come from the same instantiated `ToyDataDf` object.
            missingness_rates (float): The amount of missingness to be created, between 0.0 and 1.0 .
            repeat_imputation (int, optional): To average the MSE, perform multiple runs. The amount of runs (number of imputations) is determined by this argument. Defaults to 1.

        Returns:
            _type_: _description_
        """
        error_mse, error_mae = list(), list()
        error_raw_bias, error_percent_bias = list(), list()
        # -> Simple imputation methods <-
        if name in ['mean', 'median', 'LOCF', 'NOCB']:
            # For each missingness_rate
            for m in tqdm(missingness_rates, desc='Missingness rate'):
                mse, mae, raw_bias, percent_bias = 0, 0, 0, 0
                # Repeat, to average result
                for r in range(1, repeat_imputation+1):
                    # create missingness
                    self.create_mcar_missingness(m, verbose=False)
                    # impute
                    imputed = impute_func()
                    # add to overall MSE 
                    mse += self.mse(imputed, percent=False)
                    mae += self.mae(imputed, percent=False)
                    rb, pb = self.calc_bias(self.df.iloc[:, 2:], imputed.iloc[:, 2:], self.ind_mask[:, 2:])
                    raw_bias += rb
                    percent_bias += pb
                    
                # average MSE
                mse /= repeat_imputation
                mae /= repeat_imputation
                raw_bias /= repeat_imputation
                percent_bias /= repeat_imputation
                error_mse.append(mse)
                error_mae.append(mae)
                error_raw_bias.append(raw_bias)
                error_percent_bias.append(percent_bias)
        # -> PyPOTS imputation methods <-
        elif name in ['SAITS', 'BRITS']:
            for m in tqdm(missingness_rates, desc='Missingness rate'):
                # switch off stdout (no printing to console)
                out = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                # get imputed values
                imputed, X_intact, X, indicating_mask = impute_func(m)
                # calculate mse
                mse = cal_mse(imputed, X_intact, indicating_mask)
                mae = cal_mae(imputed, X_intact, indicating_mask)
                raw_bias, percent_bias = self.calc_bias(imputed, X_intact, indicating_mask)
                error_mse.append(mse)
                error_mae.append(mae)
                error_raw_bias.append(raw_bias)
                error_percent_bias.append(percent_bias)
                # turn on stdout again
                sys.stdout = out
        else:
            raise RuntimeError(f'Imputation name unknown. Given name: {name}')
        
        # compose error dict holding all errors
        error_dict[name] = {'mse': error_mse, 'mae': error_mae, 'raw_bias': error_raw_bias, 'percent_bias': error_percent_bias}
        # convert each list into numpy array
        for key in error_dict[name].keys():
            error_dict[name][key] = {'missingness': missingness_rates, 'value':np.array(error_dict[name][key])}
        return error_dict
    
    


    def prepare_data_pypots(self, missingness_rate, missingess_value=np.nan):
        # get rid of 'id' and 'time' column
        X = self.df.drop(['id', 'time'], axis = 1)
        X = X.to_numpy()
        # X = StandardScaler().fit_transform(X)
        # X[:,0] = self.df['time'].to_numpy() # uncomment if time should not be normalized
        X = X.reshape(self.num_samples, 50, -1)
        # create missingness
        X_intact, X, missing_mask, indicating_mask = mcar(X, missingness_rate, missingess_value) 
        X = masked_fill(X, 1 - missing_mask, np.nan)
        return X_intact, X, indicating_mask  

    def _impute_pypots(self, train_func, missingness_rate, missingess_value=np.nan, **kwargs):
        X_intact, X, indicating_mask = self.prepare_data_pypots(missingness_rate, missingess_value)
        # train the model
        model = train_func( X_intact, X, indicating_mask, **kwargs)
        # perform imputation
        imputed = model.impute(X)
        return imputed, X_intact, X, indicating_mask

    def impute_BRITS(self, missingness_rate, missingess_value=np.nan, **kwargs):
        train_func = self._train_BRITS
        return self._impute_pypots(train_func, missingness_rate, missingess_value, **kwargs)

    def _train_BRITS(self, X_intact, X, indicating_mask, log_path='./runs/brits/', **kwargs):
        n_features = X.shape[-1]
        brits = BRITS(n_steps=50, n_features=n_features, rnn_hidden_size=64, learning_rate=10e-3, epochs=20, patience=10)
        title = self.name + '_BRITS'
        brits.save_logs_to_tensorboard(saving_path=log_path, title='test')
        brits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
        return brits


    def impute_SAITS(self, missingness_rate, missingess_value=np.nan, **kwargs):
        train_func = self._train_SAITS
        return self._impute_pypots(train_func, missingness_rate, missingess_value, **kwargs)

    def _train_SAITS(self, X_intact, X, indicating_mask, log_path='./runs/saits/', **kwargs):
        """Performs imputation with SAITS

            kwargs:
            + n_steps=50 : Time steps?
            + n_features=5 : Num features in input X.
            + n_layers=2 : 
            + d_model=256 : 
            + d_inner=128 : 
            + n_head=4 : 
            + d_k=64 : 
            + d_v=64 : 
            + dropout=0 :0, Dropout value, between 0.0 and 1.0.
            + epochs=200 : Num epochs.
            + patience=30 : Patience.
        """
        
        # Model training. This is PyPOTS showtime. 💪
        n_features = X.shape[-1]
        saits = SAITS(n_steps=50, n_features=4, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.3, epochs=20, patience=30, batch_size=1024)
        title = self.name + '_SAITS'
        saits.save_logs_to_tensorboard(saving_path=log_path, title=title)
        saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
        return saits

    def prepare_mtan(self, log_path='./imputation/runs/mTAN', model_args=None, verbose=True, *args, **kwargs) -> 'tuple[DataLoader, DataLoader]':
        from toy_dataset.utils_mTAN import MTAN_ToyDataset
        # prepare dataloaders for mTAN
        dataloader_dict = self.prepare_data_mtan(**kwargs)
        # instantiate the mTAN model
        n_features = self.n_features
        self.mtan = MTAN_ToyDataset(n_features, log_path, model_args=model_args, verbose=verbose)
        return dataloader_dict

    def train_mtan(self, train_dataloader, validation_dataloader, epochs=100, **kwargs):
        # train/fit the mTAN model
        if self.mtan.epoch < epochs:
            self.mtan.args.niters = epochs
        else:
            self.mtan.args.niters += epochs
        self.mtan.train_model(train_dataloader, validation_dataloader, train_extra_epochs=epochs, **kwargs)
        return 

    def prepare_data_mtan(self, **kwargs) -> dict:        
        missingness_rate = self.artificial_missingness_rate
        missingness_value = self.artificial_missingness_value
        batch_size = kwargs.pop('batch_size', 100)
        train_dataloader, gt_train_dataloader = get_Toy_Dataloader(self.path_train, None, missingness_rate, missingness_value, batch_size=batch_size, **kwargs)
        validation_dataloader, gt_validation_dataloader = get_Toy_Dataloader(self.path_validation, None, missingness_rate, missingness_value, batch_size=batch_size, **kwargs)
        print(f'Using batch size {batch_size} for training and validation set.')
        
        dataloader_dict = {
            'train': train_dataloader,
            'train_ground_truth': gt_train_dataloader,
            'validation': validation_dataloader,
            'validation_ground_truth': gt_validation_dataloader
        }
        
        
        return dataloader_dict
    
    
    def get_dataloaders(self, **kwargs):
        missingness_rate = self.artificial_missingness_rate
        missingness_value = self.artificial_missingness_value
        batch_size = kwargs.pop('batch_size', 100)
        train_dataloader = get_Toy_Dataloader(self.path_train, None, missingness_rate, missingness_value, batch_size=batch_size, **kwargs)
        validation_dataloader = get_Toy_Dataloader(self.path_validation, None, missingness_rate, missingness_value, batch_size=batch_size, **kwargs)
        test_dataloader = get_Toy_Dataloader(self.path_test, None, missingness_rate, missingness_value, batch_size=batch_size, **kwargs)
        ground_truth_dataloader = get_Toy_Dataloader(self.path_validation, None, 0, missingness_value, batch_size=batch_size, **kwargs)
        
        print('Note:\tThe ground_truth_dataloader is based on the validation data.')
        
        return train_dataloader, validation_dataloader, test_dataloader, ground_truth_dataloader
        

    # def _train_mtan(self, train_dataloader, test_dataloader, log_path='./runs/mTAN', model_args=None, verbose=True):
    #     from toy_dataset.utils_mTAN import MTAN_ToyDataset
    #     # instantiate model
    #     n_features = self.n_features
    #     mTAN = MTAN_ToyDataset(n_features, log_path, model_args=None, verbose=True)
    #     # save logs to tensorboard is done automatically
    #     # fit
    #     mTAN.train_model(train_dataloader, test_dataloader, 10)
    #     return mTAN

    def impute_mtan(self, X_intact:np.ndarray, X:np.ndarray, ind_mask:np.ndarray, time_pts:np.ndarray):
        if not hasattr(self, 'mtan'):
            raise RuntimeError(r'The object does not have .mtan as an attribute. Please first run `self.prepare_mtan(_)`')

        self.mtan.impute()




        

def get_sample_train_ground_truth(dataloader:DataLoader, dataset:ToyDataDf, sample_num:int):
    """Get a single sample. The training sample with the corresponding ground truth sample.

    Args:
        dataloader (DataLoader): Train dataloader.
        dataset (ToyDataDf): Dataset containing ground truth.
        sample_num (int): Number of sample, that should be returned.

    Returns:
        list[torch.Tensor, torch.Tensor]: ground_truth, training_sample
    """
    import itertools
    # batch from dataloader (e.g. for training)
    batch_size = dataloader.batch_size
    batch_num = int(np.floor(sample_num/batch_size))
    batch = next(itertools.islice(dataloader, batch_num, None))
    sample_in_batch = sample_num % batch_size
    training_sample = batch[sample_in_batch, ...]
    
    # batch from dataset containing ground truth
    X_intact, X, ind_mask, time_pts, id = dataset.get_sample(sample_num)
    ground_truth = X_intact

    return ground_truth, training_sample


def get_batch_train_ground_truth(trainloader:DataLoader, ground_truth_loader:DataLoader, batch_num:int):
    """Get a single sample. The training sample with the corresponding ground truth sample.

    Args:
        dataloader (DataLoader): Train dataloader.
        dataset (DataLoader): Dataset containing ground truth.
        batch_num (int): Number of batch, that should be returned.

    Returns:
        list[torch.Tensor, torch.Tensor]: ground_truth, training_batch
    """
    import itertools

    training_batch = next(itertools.islice(trainloader, batch_num, None))
    ground_truth = next(itertools.islice(ground_truth_loader, batch_num, None))

    return ground_truth, training_batch





def calculate_cr_aw(hetvae:'HETVAE', dataloader, gt_dataloader, num_samples=100, sample_tp=0.9, quantile=0.68) -> 'dict[str, dict]':
    """Returns the Coverage Ratio (CR) and Average width (AW) between imputed values and the ground truth.
    
    Note: This function should be adapted, so that it just takes a dataloader that loads the predictions in some way.
    

    Args:
        hetvae (HETVAE): HeTVAE imputation model.
        dataloader (DataLoader): The train or validation dataloader for imputing values.
        gt_dataloader (DataLoader): Data loader for same data as in `dataloader`, but contains the ground truth.
        num_samples (int, optional): Num samples to be drawn during imputation. Defaults to 100.
        sample_tp (float, optional): Subsampling for HeTVAE during imputation. Defaults to 0.9.
        quantile (float, optional): The inner quantile of imputed values to be calculated. Defaults to 0.68.

    Returns:
        np.array, np.array: coverages and widths PER DIMENSION.
    """
    dim = hetvae.n_features
    base_dict = {'coverage_ratio': list(), 'average_width': list()}
    error_dict = {'subsample': deepcopy(base_dict), 'recon': deepcopy(base_dict), 'gt': deepcopy(base_dict)}
    # coverages = list()
    # widths = list()
    # we iterate through the batches
    for train_batch, gt_batch in tqdm(zip(dataloader, gt_dataloader), total=len(dataloader)):

        # get the predictions
        pred_mean, preds, quantile_low, quantile_high, mask_dict = hetvae.impute(train_batch, num_samples, sample_tp=sample_tp, quantile=quantile)
        gt = gt_batch[:,:,:dim].numpy()
        
        # sanity check that all masks are correct
        mask_sub = mask_dict['subsample']
        mask_recon = mask_dict['recon']
        mask_gt = mask_dict['gt']
        assert(((mask_sub+mask_recon+mask_gt) == 1).all()), 'There were values above 1, when adding.'
        assert(((mask_sub*mask_recon*mask_gt)==0).all()), 'There were values != 0, when multiplying'
        
        # iterate over all three data cases 'subsample', 'recon', 'gt'
        for data_case in error_dict.keys():
            # get the corresponding mask
            mask = mask_dict[data_case].cpu().numpy()
            gt_masked = gt * mask
            # coverage rate
            cov = ((quantile_low <= gt_masked) * (gt_masked <= quantile_high)).mean(axis=1) # averaging per sample, remaining samples and features
            # widths
            width = np.abs((quantile_high - quantile_low)*mask).mean(axis=1)
            # add to the previous values
            error_dict[data_case]['coverage_ratio'].append(cov) 
            error_dict[data_case]['average_width'].append(width)
            
    # we want mean error per variable, so we iterate through datacase in error_dict
    for data_case in error_dict.keys():
        # iterate through all 
        for metric_name in error_dict[data_case].keys():
            error_dict[data_case][metric_name] = np.concatenate(error_dict[data_case][metric_name]).mean(axis=0)
            
        
        

    # # average over the samples
    # coverages = np.concatenate(coverages).mean(axis=0)
    # widths = np.concatenate(widths).mean(axis=0)
    
    
    return error_dict


def calculate_bias(hetvae:'HETVAE', dataloader, gt_dataloader, num_samples=100, sample_tp=0.9) -> 'dict[str, dict]':
    """Returns the Percent Bias (PB), which stems from the Raw Bias (RB).
    
    Note: This function should be adapted, so that it just takes a dataloader that loads the predictions in some way.

    Args:
        hetvae (HETVAE): Trained HETVAE model.
        dataloader (DataLoader): Train or Validation loader.
        gt_dataloader (DataLoader): Same data as dataloader but without missingness.
        num_samples (int, optional): Number of samples to be drawn by HeTVAE during imputation. Defaults to 100.
        sample_tp (float, optional): How many points should be subsampled by HetTVAE during imputation process.. Defaults to 0.9.

    Returns:
        dict: Error_dict, where error_dict[data_case][metric_name] is a np.array with error values for each variable.
    """
    dim = hetvae.n_features
    base_dict = {'percent_bias': list(), 'raw_bias': list()}
    error_dict = {'subsample': deepcopy(base_dict), 'recon': deepcopy(base_dict), 'gt': deepcopy(base_dict)}
    
    for train_batch, gt_batch in tqdm(zip(dataloader, gt_dataloader), total=len(dataloader)):
        # get the predictions and ground truth
        pred_mean, preds, quantile_low, quantile_high, mask_dict = hetvae.impute(train_batch, num_samples, sample_tp=sample_tp)
        gt = gt_batch[:,:,:dim].numpy()
        # iterate over all three data cases 'subsample', 'recon', 'gt'
        for data_case in error_dict.keys():
            # get the corresponding mask
            mask = mask_dict[data_case].cpu().numpy()
            # calc the rb and pb for this batch
            rb, pb = ToyDataDf.calc_bias(gt, pred_mean, mask, keep_vars=True) # one dimensional, as many entries as there are vars
            # add to the previous values
            error_dict[data_case]['raw_bias'].append(rb) 
            error_dict[data_case]['percent_bias'].append(pb)
    
    
    # # average over all samples
    # raw_bias = np.stack(raw_bias, axis=0).mean(axis=0)
    # percent_bias = np.stack(percent_bias, axis=0).mean(axis=0)
    
    # we want mean error per variable, so we iterate through datacase in error_dict
    for data_case in error_dict.keys():
        # iterate through all 
        for metric_name in error_dict[data_case].keys():
            error_dict[data_case][metric_name] = np.stack(error_dict[data_case][metric_name], axis=0).mean(axis=0)
    
    return error_dict
