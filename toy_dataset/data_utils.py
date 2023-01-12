"""Utility functions for Toydataset / Synthetic dataset.
    Multivarite time series.
"""
import os 
import sys
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
        X_intact, X, missing_mask, indicating_mask = self.missingness(x, 
                                                                        self.missingness_rate, 
                                                                        self.missingness_value)
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

    
def get_Toy_Dataloader(path, missingness=None, missingness_rate=0.3, missingness_value=-1, batch_size=1, shuffle=False, **kwargs) -> DataLoader:
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

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

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
        df = self.df.copy()
        # create missingness in data
        df_intact, df_mis, miss_mask, ind_mask = pc.mcar(df.iloc[:,2:].to_numpy(), missingness_rate, missingness_value)
        ind_mask = (ind_mask==0) * 1
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
        # -> Simple imputation methods <-
        if name in ['mean', 'median', 'LOCF', 'NOCB']:
            # For each missingness_rate
            for m in tqdm(missingness_rates, desc='Missingness rate'):
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
                error.append(mse)
                # turn on stdout again
                sys.stdout = out
        else:
            raise RuntimeError(f'Imputation name unknown. Given name: {name}')
        
        error_dict[name] = error
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
        saits = SAITS(n_steps=50, n_features=5, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.3, epochs=50, patience=30)
        title = self.name + '_SAITS'
        saits.save_logs_to_tensorboard(saving_path=log_path, title=title)
        saits.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
        return saits

    def prepare_mtan(self, log_path='./imputation/runs/mTAN', model_args=None, verbose=True, *args, **kwargs) -> 'tuple[DataLoader, DataLoader]':
        from toy_dataset.utils_mTAN import MTAN_ToyDataset
        # prepare dataloaders for mTAN
        train_dataloader, validation_dataloader = self.prepare_data_mtan(**kwargs)
        # instantiate the mTAN model
        n_features = self.n_features
        self.mtan = MTAN_ToyDataset(n_features, log_path, model_args=model_args, verbose=verbose)
        return train_dataloader, validation_dataloader

    def train_mtan(self, train_dataloader, validation_dataloader, epochs=100, **kwargs):
        # train/fit the mTAN model
        if self.mtan.epoch < epochs:
            self.mtan.args.niters = epochs
        else:
            self.mtan.args.niters += epochs
        self.mtan.train_model(train_dataloader, validation_dataloader, train_extra_epochs=epochs, **kwargs)
        return

    def prepare_data_mtan(self, **kwargs) -> 'tuple[DataLoader, DataLoader]':        
        missingness_rate = self.artificial_missingness_rate
        missingness_value = self.artificial_missingness_value
        batch_size = kwargs.pop('batch_size', 100)
        train_dataloader = get_Toy_Dataloader(self.path_train, None, missingness_rate, missingness_value, batch_size=batch_size, **kwargs)
        validation_dataloader = get_Toy_Dataloader(self.path_validation, None, missingness_rate, missingness_value, batch_size=batch_size, **kwargs)
        print(f'Using batch size {batch_size} for training and validation set.')
        
        return train_dataloader, validation_dataloader

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




        

def get_complete_batch(dataloader:DataLoader, dataset:ToyDataDf, sample_num:int):
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

    return ground_truth, training_sample, X









