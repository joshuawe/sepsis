"""Generating synthetic time series data with four variables representing:
+ Noise
+ Trend (+ Noise)
+ Seasonality (+ Noise )
+ Trend + Seasonality (+ Noise)

The generated data will be saved in the same way as the data created above. It will have the columns `'id', 'time', 'Noise', 'Trend', 'Seasonality', 'Trend + Season'`. During generation of data it will be handled as a `pandas.DataFrame` object and then saved as a `*.csv.gz` file under the specified location.

The four parts of data:
+ The **noise** part is by sampling from a normal distribution with $\mu=0$ and $\sigma=1$.
+ The **trend** part consists of $m \cdot x(t) + n + \text{noise}$ , where $m,n$ are sampled from the same distribution as the noise.
+ The **seasonality** part has its frequency and amplitude drawn form a uniform distribution and then computes to $\text{amplitude} \cdot \sin{(x(t) \cdot \text{frequency})} + \text{noise} \cdot \epsilon$, where $\epsilon$ is drawn from the standard normal distribution.
+ The **trend+seasonality** part is computed via $m \cdot \text{trend} + n + \text{seasonality} + \text{noise} \cdot \epsilon$, where $m, n, \epsilon$ have been drawn independently from the previous parts.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def generate_ts_sample(series_length=100):
    """ There are exactly 4 cases: noise, trend, seasonality, trend + seasonality
    
    Args:
        series_length (int): The length of the time series.

    Returns:
        np.arrays: time, noise, trend, seasonality, trend_season
    """
    # for each time series: noise, trend, seasonality, trend + seasonality
    time = np.arange(series_length) + np.random.randint(-100,100) # add a random shift
    # noise
    noise = np.random.randn(series_length)
    # trend
    m = np.random.rand()  # slope
    n = np.random.rand()  # y-axis at x=0
    trend = m * time + n + noise
    # seasonality
    frequency = np.random.uniform(0.5,1)
    amplitude = np.random.uniform(1, 10)
    seasonality = amplitude * np.sin(time * frequency) + noise * np.random.rand()
    # trend + seasonality
    m = np.random.uniform(0.1,3)   # slope
    n = np.random.rand() * 100  # y-axis at x=0
    mu = 0.8
    sigma = 0.1
    trend_season = m * trend + n + seasonality  + np.random.normal(mu, sigma, series_length) * noise

    return time, noise, trend, seasonality, trend_season
    


def genereate_ts_dataset(samples, series_length, save_path=None):
    """Creates Time series dataset with variables: id, time, noise, trend, seasonality, trend + seasonality.

    Args:
        samples (int): The number of individual time series, that comprise the dataset.
        series_length (int): The length of every series.
        save_path (str): The path (folder + filename + *.csv.gz)

    Returns:
        pd.DataFrame: The dataframe containing all the time series.
    """
    columns = ['id', 'time', 'Noise', 'Trend', 'Seasonality', 'Trend + Season']
    df = pd.DataFrame(columns=columns)
    time_series_list = list()
    time_series_list.append(df)
    for i in tqdm(range(samples)):
        # create the time series
        time, noise, trend, seasonality, trend_season = generate_ts_sample(series_length)
        time = np.arange(series_length)
        data = np.stack((time, noise, trend, seasonality, trend_season), axis=1)
        # create the corresponding id
        id = 'id_' + str(i)
        # convert into dataframe
        df_temp = pd.DataFrame(data=data ,columns=columns[1:])
        df_temp['id'] = id
        # add to list
        time_series_list.append(df_temp)
    # combine
    df = pd.concat(time_series_list, ignore_index=True)

    if save_path is not None:
        print(f'Saving dataset under {save_path}\nMake sure to have a *.csv.gz file ending.')
        df.to_csv(save_path, sep=",", compression=None, index=False)
    return df


def visualize_new_sample():
    """Visualizes a new time series sample that is created by calling the function. So, calling the function a few time will give some examples of what the time series could look like.
    """
    import matplotlib.pyplot as plt
    time, noise, trend, seasonality, trend_season = generate_ts_sample()
    time = np.arange(len(time))

    fig = plt.figure()
    fig.add_subplot(221)
    plt.plot(time, noise)
    plt.title('Noise')
    plt.xlabel('time t')
    plt.ylabel('y')
    fig.add_subplot(222)
    plt.title('Trend')
    plt.xlabel('time t')
    plt.ylabel('y')
    plt.plot(time, trend)
    fig.add_subplot(223)
    plt.xlabel('time t')
    plt.ylabel('y')
    plt.title('Seasonality')
    plt.plot(time, seasonality)
    fig.add_subplot(224)
    plt.title('Trend + Seasonality')
    plt.xlabel('time t')
    plt.ylabel('y')
    plt.plot(time, trend_season)
    plt.tight_layout()
    plt.show()
    # print('Note, that in the proper dataset the time always starts at 0, basically a shift of the x-axis.')