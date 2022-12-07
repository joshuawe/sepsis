"""Utility functions for Toydataset / Synthetic dataset.
    Multivarite time series.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pycorruptor as pc
from tqdm import tqdm

datasets_dict = {'toydataset_small': {
                                'descr': 'A very small and simple dataset, that is used for initial testing, not even a proper toy data set, very small.',
                                'path': '/home2/joshua.wendland/Documents/sepsis/toy_dataset/synthetic_ts_1/synthetic_ts_test_data_eav.csv.gz'
                                }
}

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
            pass
        else:
            self.missingness_rate = 0
        
    
        self.missingness = pc.mcar
        self.missingness_rate = missingness_rate
        self.missingness_value = missingness_value
        if self.missingness_rate in [0,1]: print(f"Attention, missingness_rate = {self.missingness_rate} !")
        # Read in csv
        df = pd.read_csv(path, compression=None)
        self.input_dim = len(df.columns) - 2  # no 'id' and 'time' column. What about 'time' column?
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
        X_intact, X, missing_mask, indicating_mask = self.missingness(x, 
                                                                        self.missingness_rate, 
                                                                        self.missingness_value)
        Y = X_intact

        # concatenate all information into X
        # time = time.unsqueeze(1)
        # X_concat = torch.concatenate((X, indicating_mask, time), dim=1)
        
        # sample = X_concat, x_ids, Y
        # X = torch.tensor(X.clone().detach(), dtype=torch.float32)
        X = X.type(torch.float32)
        missing_mask = missing_mask.type(torch.float32)
        time = time.type(torch.float32)
        Y = Y.type(torch.float32)
        sample = X, missing_mask, time, Y
        sample = torch.concatenate((X_intact, missing_mask, time.unsqueeze(1)), dim=1)
        sample = sample.type(torch.float32)
        return sample


    def __len__(self):
        """The number of samples (time series) available in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.n_samples

    
def get_Toy_Dataloader(path, missingness=None, missingness_rate=0.3, missingness_value=-1, batch_size=1, shuffle=False):
        """Wrapper function for the Toydataset in order to get the DataLoader directly.

        Args:
            path (str): Path to dataset.
            missingness (str, optional): Check out ToyDataset class. Defaults to None.
            missingness_rate (float, optional): Defaults to 0.3.
            missingness_value (float, optional): Defaults to -1.
            batch_size (int, optional):  Defaults to 1.
            shuffle (bool, optional): Shuffle the samples in DataLoader. Defaults to False.

        Returns:
            torch.DataLoader: The Dataloader for the Toy Dataset.
        """
        # create dataset
        dataset = ToyDataset(path, missingness=missingness, missingness_rate=missingness_rate, missingness_value=missingness_value)

        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        return DataLoader

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
    def __init__(self, path) -> pd.DataFrame:
        """Fetches the Toy Dataset from `path` and returns a `pandas.DataFrame`.

        Args:
            path (str): Path to dataset.

        Returns:
            pandas.DataFrame: The Dataset.
        """
        self.df = pd.read_csv(path, compression=None)
        self.artificial_missingness = None
        self.name = 'Toydataset'
        return

    def create_mcar_missingness(self, missingness_rate, missingness_value=np.nan, verbose=False):
        """Creates MCAR missingness on `self.df`. The first two columns are not included in the missingness process, as they are assumed to be 'id' and 'time'. The dataset with missingness and the corresponding mask are saved in `self.df_mis` and `self.ind_mask`. The shape of `self.df`, `self.df_mis` and `self.ind_mask` is the same.

        Args:
            missingness_rate (float): The amount of additionally missing data. Between 0.0 and 1.0 .
            missingness_value (float, optional): The value the missing datapoints are to be assigned. Defaults to np.nan.
            verbose (bool, optional): Print information to console. Defaults to False.
        """
        df = self.df.copy()
        # create missingness in data
        df_intact, df_mis, miss_mask, ind_mask = pc.mcar(df.iloc[:,2:].to_numpy(), missingness_rate, missingness_value)
        num_values = miss_mask.size
        # add the missingdata back into df
        df.iloc[:,2:] = pd.DataFrame(df_mis, columns=df.columns[2:])
        # Add the two discarded columns to the miss_mask and ind_mask
        ones = np.ones((miss_mask.shape[0],2))
        miss_mask = np.concatenate((ones, miss_mask), axis=1)
        ind_mask = np.concatenate((ones, ind_mask), axis=1)
        # save data to class
        self.df_mis = df
        self.ind_mask = ind_mask
        self.artificial_missingness = 'mcar'
        if verbose:
            print(f'--\nCreated MCAR missing data, but without missingness in columns {df.columns[:2]}')
            print(f'missingness_rate: {missingness_rate},\tmissingness_value: {missingness_value}')
            num_mis = df.isna().sum().sum()
            print(f'Missing values: {num_mis} out of {num_values} ({num_mis/num_values:.1%}) (!excluding aforementioned columns)')
            print(f'Data values in entire dataframe is {df.size} (shape: {df.shape})')
        return

    def impute_mean(self):
        """Perform mean imputation on the dataset. Only works if missingness has been created already.

        Returns:
            pd.DataFrame: The dataset with mean imputation.
        """
        imputation_func = pd.DataFrame.mean
        kwargs = {'axis':0, 'numeric_only':True}
        return self._base_impute(imputation_func, **kwargs)

    def impute_median(self):
        """Perform median imputation on the dataset. Only works if missingness has been created already.

        Returns:
            pd.DataFrame: The dataset with median imputation.
        """
        imputation_func = pd.DataFrame.median
        kwargs = {'axis':0, 'numeric_only':True}
        return self._base_impute(imputation_func, **kwargs)

    def impute_LOCF(self):
        """Perform LOCF (last observation carried forward) imputation on the dataset. Only works if missingness has been created already.

        Returns:
            pd.DataFrame: The dataset with LOCF imputation.
        """
        imputation_func = pd.DataFrame.fillna
        kwargs = {'method':'ffill'}
        return self._base_impute(imputation_func, **kwargs)

    def impute_NOCB(self):
        """Perform NOCB (next observation carried backwards) imputation on the dataset. Only works if missingness has been created already.

        Returns:
            pd.DataFrame: The dataset with NOCB imputation.
        """
        imputation_func = pd.DataFrame.fillna
        kwargs = {'method':'backfill'}
        return self._base_impute(imputation_func, **kwargs)

    def _base_impute(self, imputation_func, **kwargs):
        """Base function for simple imputing methods. Groups the time series in the dataset by 'id' and the performs the corresponding imputation method that was passed as argument in `imputation_func`.

        Args:
            imputation_func (class function): The *pandas.DataFrame* function that performes desired imputation method.
            kwargs: The kwargs that are passes as arguments to the `imputation_func`.

        Raises:
            RuntimeError: Only, if no missingness has been created already.

        Returns:
            pd.DataFrame: Dataframe with the imputed values.
        """
        if self.artificial_missingness is None:
            raise RuntimeError('First create missingness.')
        df = self.df_mis.copy()
        # get all time series IDs
        IDs = df['id'].unique()
        # Cycle through each time series ID
        for id in IDs:
            # get only the time series with corresponding ID
            ts = df.loc[df['id'] == id]
            # get the imputation value for each column
            impu = imputation_func(ts, **kwargs)
            # replace nan values with imputed values
            df.loc[df['id'] == id] = ts.fillna(impu)
        return df

    def mse(self, imputed_df:pd.DataFrame):
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
        sum = mse.sum().sum()
        # only consider values that had been missing before
        #    Not necessary here, but good practice, as later on it will be necessary.
        mse *= self.ind_mask[:, 2:]
        mse =  mse.sum().sum()
        mse /= self.ind_mask[:, 2:].sum().sum()

        return mse

    def get_mse_impute(self, name, error_dict, impute_func, missingness_rates, repeat_imputation=1):
        """Calculates the (average) MSE for a given imputation method passes as the argument `impute_func`.

        Args:
            name (str): Name of imputation method.
            error_dict (dict): A dict into which the new errors can be entered via `error_dict[name] = mse_imputation_method`.
            impute_func (class function): The imputation function to be executed. It should come from the same instantiated `ToyDataDf` object.
            missingness_rates (float): The amount of missingness to be created, between 0.0 and 1.0 .
            repeat_imputation (int, optional): To average the MSE, perform multiple runs. The amount of runs (number of imputations) is determined by this argument. Defaults to 1.

        Returns:
            _type_: _description_
        """
        error = list()
        # For each missingness_rate
        for m in tqdm(missingness_rates):
            mse = 0
            # Repeat, to average result
            for r in range(1, repeat_imputation+1):
                # create missingness
                self.create_mcar_missingness(m, verbose=False)
                # impute
                imputed = impute_func()
                # add to overall MSE 
                mse += self.mse(imputed)
            # average MSE
            mse /= repeat_imputation
            error.append(mse)
        
        error_dict[name] = error
        return error_dict

    


        







