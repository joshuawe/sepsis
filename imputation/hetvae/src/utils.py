# pylint: disable=E1101
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt


def union_time(data_loader, classif=False):
    # return all time points of the training data in the data loader, sorted in ascending order.
    tu = []
    # iterate through each batch
    for batch in data_loader:
        if classif:
            batch = batch[0]
        # take all time points
        tp = batch[:, :, -1].numpy().flatten()
        # cycle through each time point from batch
        for val in tp:
            # add all time points which are not stored in tu yet
            if val not in tu:
                tu.append(val)
    # sort all tp
    tu.sort()
    return torch.from_numpy(np.array(tu))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask


def mog_log_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2.0 * np.pi])).float().to(x.device)
    const = torch.log(const)
    const2 = torch.from_numpy(np.array([mean.size(0)])).float().to(x.device)
    loglik = -0.5 * (const + logvar + (x - mean) ** 2.0 / torch.exp(logvar)) * mask
    return torch.logsumexp(loglik - torch.log(const2), 0)


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.0
    lstd2 = lv2 / 2.0
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2)) - 0.5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def mean_absolute_error(orig, pred, mask):
    error = torch.abs(orig - pred)
    error = error * mask
    return error.sum() / mask.sum()


def evaluate_hetvae(
    net,
    dim,
    train_loader,
    ground_truth_loader = None,
    sample_tp=0.5,
    shuffle=False,
    k_iwae=5,
    device='cuda',
):
    torch.manual_seed(seed=0)
    np.random.seed(seed=0)
    val_loss, train_n = 0, 0
    avg_loglik, mse, mae = 0, 0, 0
    mean_mae, mean_mse = 0, 0
    with torch.no_grad():
        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            subsampled_mask = subsample_timepoints(
                train_batch[:, :, dim:2 * dim].clone(),
                sample_tp,
                shuffle=shuffle,
            )
            recon_mask = train_batch[:, :, dim:2 * dim] - subsampled_mask
            context_y = torch.cat((
                train_batch[:, :, :dim] * subsampled_mask, subsampled_mask
            ), -1)
            loss_info = net.compute_unsupervised_loss(
                train_batch[:, :, -1],
                context_y,
                train_batch[:, :, -1],
                torch.cat((
                    train_batch[:, :, :dim] * recon_mask, recon_mask
                ), -1),
                num_samples=k_iwae,
            )
            num_context_points = recon_mask.sum().item()
            val_loss += loss_info.composite_loss.item() * num_context_points
            mse += loss_info.mse * num_context_points
            mae += loss_info.mae * num_context_points
            mean_mse += loss_info.mean_mse * num_context_points
            mean_mae += loss_info.mean_mae * num_context_points
            avg_loglik += loss_info.mogloglik * num_context_points
            train_n += num_context_points
    print(
        'nll: {:.4f}, mse: {:.4f}, mae: {:.4f}, '
        'mean_mse: {:.4f}, mean_mae: {:.4f}'.format(
            - avg_loglik / train_n,
            mse / train_n,
            mae / train_n,
            mean_mse / train_n,
            mean_mae / train_n,
        ), flush=True
    )
    return ( val_loss/train_n, - avg_loglik/train_n, mse/train_n, mae/train_n, mean_mse/train_n, mean_mae/train_n)


def get_mimiciii_data(batch_size, test_batch_size=5, filter_anomalies=True):
    input_dim = 12
    x = np.load("../../neuraltimeseries/Dataset/final_input3.npy")
    x = x[:, :25]
    x = np.transpose(x, (0, 2, 1))
    observed_vals, observed_mask, observed_tp = (
        x[:, :, :input_dim],
        x[:, :, input_dim: 2 * input_dim],
        x[:, :, -1],
    )
    print(observed_vals.shape, observed_mask.shape, observed_tp.shape)

    if np.max(observed_tp) != 0.0:
        observed_tp = observed_tp / np.max(observed_tp)

    if filter_anomalies:
        data_mean, data_std = [], []
        var_dict = {}
        hth = []
        lth = []
        for i in range(input_dim):
            var_dict[i] = []
        for i in range(observed_vals.shape[0]):
            for j in range(input_dim):
                indices = np.where(observed_mask[i, :, j] > 0)[0]
                var_dict[j] += observed_vals[i, indices, j].tolist()

        for i in range(input_dim):
            th1 = np.quantile(var_dict[i], 0.001)
            th2 = np.quantile(var_dict[i], 0.9995)
            hth.append(th2)
            lth.append(th1)
            temp = []
            for val in var_dict[i]:
                if val <= th2 and val >= th1:
                    temp.append(val)
            if len(np.unique(temp)) > 10:
                data_mean.append(np.mean(temp))
                data_std.append(np.std(temp))
            else:
                data_mean.append(0)
                data_std.append(1)

        # normalizing
        observed_vals = (observed_vals - data_mean) / data_std
        observed_vals[observed_mask == 0] = 0
    else:
        for k in range(input_dim):
            data_min, data_max = float("inf"), 0.0
            for i in range(observed_vals.shape[0]):
                for j in range(observed_vals.shape[1]):
                    if observed_mask[i, j, k]:
                        data_min = min(data_min, observed_vals[i, j, k])
                        data_max = max(data_max, observed_vals[i, j, k])
            # print(data_min, data_max)
            if data_max == 0:
                data_max = 1
            observed_vals[:, :, k] = (observed_vals[:, :, k] - data_min) / data_max
        # set masked out elements back to zero
        observed_vals[observed_mask == 0] = 0

    total_dataset = np.concatenate(
        (observed_vals, observed_mask, observed_tp[:, :, None]), -1)
    print(total_dataset.shape)
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(
        total_dataset, train_size=0.8, random_state=42, shuffle=True
    )
    # for interpolation, we dont need a non-overlapping validation set as
    # we can condition on different set of time points from same set to
    # create distinct examples
    _, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=11, shuffle=True
    )
    print(train_data.shape, val_data.shape, test_data.shape)
    train_data = torch.from_numpy(train_data).float()
    val_data = torch.from_numpy(val_data).float()
    test_data = torch.from_numpy(test_data).float()

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=100, shuffle=False)

    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": input_dim,
    }
    return data_objects


def get_physionet_data(batch_size, test_batch_size=5):
    input_dim = 41
    # data = np.load("../data/physionet_compressed.npz")
    data = np.load("/home2/joshua.wendland/Documents/sepsis/imputation/hetvae/data/physionet_compressed.npz")
    train_data, test_data = data['train'], data['test']
    # for interpolation, we dont need a non-overlapping validation set as
    # we can condition on different set of time points from same dataset to
    # create a distinct example
    _, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=11, shuffle=True
    )
    print('train_data.shape', train_data.shape, '\ttest_data.shape', test_data.shape)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=100, shuffle=False)

    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": input_dim,
    }
    return data_objects


def get_synthetic_data(
    args,
    alpha=120.0,
    seed=0,
    ref_points=10,
    total_points=51,
    add_noise=True,
):
    np.random.seed(seed)
    ground_truth, ground_truth_tp = [], []
    observed_values = []
    for _ in range(args.n):
        key_values = np.random.randn(ref_points)
        key_points = np.linspace(0, 1, ref_points)
        query_points = np.linspace(0, 1, total_points)
        weights = np.exp(-alpha * (
            np.expand_dims(query_points, 1) - np.expand_dims(key_points, 0)
        ) ** 2)
        weights /= weights.sum(1, keepdims=True)
        query_values = np.dot(weights, key_values)
        ground_truth.append(query_values)
        if add_noise:
            noisy_query_values = query_values + 0.1 * np.random.randn(total_points)
        observed_values.append(noisy_query_values)
        ground_truth_tp.append(query_points)

    observed_values = np.array(observed_values)
    ground_truth = np.array(ground_truth)
    ground_truth_tp = np.array(ground_truth_tp)
    observed_mask = np.ones_like(observed_values)

    observed_values = np.concatenate(
        (
            np.expand_dims(observed_values, axis=2),
            np.expand_dims(observed_mask, axis=2),
            np.expand_dims(ground_truth_tp, axis=2),
        ),
        axis=2,
    )
    print(observed_values.shape)
    train_data, test_data = model_selection.train_test_split(
        observed_values, train_size=0.8, random_state=42, shuffle=True
    )
    _, ground_truth_test = model_selection.train_test_split(
        ground_truth, train_size=0.8, random_state=42, shuffle=True
    )
    _, val_data = model_selection.train_test_split(
        train_data, train_size=0.8, random_state=42, shuffle=True
    )
    print(train_data.shape, val_data.shape, test_data.shape)
    train_dataloader = DataLoader(
        torch.from_numpy(train_data).float(), batch_size=args.batch_size, shuffle=False
    )
    val_dataloader = DataLoader(
        torch.from_numpy(val_data).float(), batch_size=args.batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        torch.from_numpy(test_data).float(), batch_size=5, shuffle=False
    )

    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "val_dataloader": val_dataloader,
        "input_dim": 1,
        "ground_truth": ground_truth_test,
    }
    return data_objects


def get_toydata(batch_size):
    from toy_dataset import data_utils

    name = 'toydataset_50000'
    path = data_utils.datasets_dict[name]
    dataset = data_utils.ToyDataDf(path)
    dataset.create_mcar_missingness(0.3, -1)
    model_args = '--niters 2000 --lr 0.0001 --batch-size 128 --rec-hidden 16 --latent-dim 64 --embed-time 128 --enc-num-heads 1 --num-ref-points 16 --n 2000 --dataset toy --seed 0 --norm --sample-tp 0.5 --k-iwae 1'.split()
    train_dataloader, validation_dataloader = dataset.prepare_mtan(model_args=model_args, batch_size=batch_size)

    print('Note: Validation and test dataloader are the same!')

    data_objects = {
        "train_dataloader": train_dataloader,
        "test_dataloader": validation_dataloader,
        "val_dataloader": validation_dataloader,
        "input_dim": 4,
        "ground_truth": None,
    }
    return data_objects


def subsample_timepoints(mask, percentage_tp_to_sample=None, shuffle=False):
    # Subsample percentage of points from each time series
    if not shuffle:
        seed = 0
        np.random.seed(seed)
    else:
        seed = np.random.randint(0, 100000)
        np.random.seed(seed)
    for i in range(mask.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(
            np.random.choice(non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.
    return mask


def visualize_sample(batch, pred_mean, quantile_low=None, quantile_high=None, ground_truth=None, print_dims=None, sample=None, title=''):
    """*(Author: Josh)* Visualizing a sample with matplotlib. It will display max 4 features in a 2x2 figure grid. If `sample is None`, a random sample will be drawn from the batch.

    Args:
        batch (torch.Tensor): A train batch.
        pred_mean (torch.Tensor): The prediction of the NN based on batch.
        quantile_low (np.array): Lower quantile of distribution to be displayed.
        quantile_high (np.array): Higher quantile of distribution to be displayed.
        ground_truth (torch.Tensor, optional): The ground truth. Defaults to None.
        dim (int): Number of features. Can be deduced automatically.
        sample (int, optional): The sample to be displayed from batch. Defaults to None.
        title (str, optional): Beginning of figure title. Defaults to ''.
    """
    # get num features from batch shape
    dim = (batch.shape[-1] - 1) / 2
    assert(dim % 1 == 0), f'dim should be an integer, instead dim = {dim}'
    dim = int(dim)
    # define print dims
    print_dims = print_dims if print_dims is not None else dim
    if sample is None:
        sample = np.random.randint(low=0, high=batch.shape[0])
        title += 'Random '
        
    fig = plt.figure(figsize=(9, 2*dim))
    # only use the required sample from the batches
    batch = batch[sample, :, :]
    # only use time points, where the t>0 to clip out padding
    time_points = (batch[:,-1] != 0)
    batch = batch[time_points] 
    pred_mean = pred_mean[sample, :, :]
    pred_mean = pred_mean[time_points] # only the time points we want
    if quantile_low is not None and quantile_high is not None:
        quantile_low  = quantile_low[sample, :, :]
        quantile_high = quantile_high[sample, :, :]
        quantile_low  = quantile_low[time_points]
        quantile_high = quantile_high[time_points]
    if ground_truth is not None:
        ground_truth = ground_truth[sample, :, :]
        ground_truth = ground_truth[time_points]
    
    # the actual values of the time points (time_points is a boolean array)
    x_time = batch[:, -1]


    for feature in range(print_dims):
        ax = fig.add_subplot(dim,1,feature+1)
        # x_predicted = pred_mean[0, sample, :, feature]
        x_predicted = pred_mean[:, feature]
        x_observed = batch[:,feature].cpu()
        x_mask = batch[:, dim+feature].cpu()
        # 1) plot prediction
        plt.plot(x_time, x_predicted, alpha=0.7, marker='o', label='predicted', c='C1')
        # 2) plot observed
        x = np.array(x_observed)
        x[x_mask==0] = np.nan
        if ground_truth is None:
            plt.scatter(x_time, x, alpha=1, marker='o', label='observed', facecolor='None', edgecolor='black')
        else:
            plt.plot(x_time, x, alpha=1, marker='o', label='observed', markerfacecolor='None', markeredgecolor='black', color='black')
        # 3) plot masked: only plot, if ground_truth is not there
        if ground_truth is None: 
            # plot masked 
            x_masked = np.array(x_observed)
            x_masked[x_mask==1] = np.nan
            plt.scatter(x_time, x_masked, marker='o', alpha=0.5, label='masked', c='C0')
        # 4) plot ground truth
        if ground_truth is not None:
            plt.plot(x_time, ground_truth[:, feature], alpha=1, c='C3', ls='-', label='ground truth')
        # 5) plot quantiles (uncertainty)
        if quantile_low is not None and quantile_high is not None:
            ql = quantile_low[:, feature]
            qh = quantile_high[:, feature] 
            plt.fill_between(x_time, ql, qh, alpha=0.45, facecolor='#65c9f7', interpolate=True, label='uncertainty')
            # plt.vlines(x_time, ql, qh, color='grey', capstyle='round', linewidths=3, alpha=0.3)
        # set labels and limits
        min = np.min((x_predicted.min(), x_observed[x_observed>-1].min())) - 0.1
        max = np.max((x_predicted.max(), x_observed.max())) + 0.1
        ax.set(xlabel='time', ylabel=f'$X_{feature}$', ylim=(min, max))
        ax.set_xticks(np.arange(0,50,5), minor=True)

        if feature==0:
            plt.legend(loc='lower center', ncol=5)


    mse = mean_squared_error(batch[:,:dim], pred_mean[:,:], batch[:, dim:2*dim])
    mae = mean_absolute_error(batch[:,:dim], pred_mean[:,:], batch[:, dim:2*dim])
    plt.legend(loc='lower center', ncol=5)
    fig.suptitle(title + f'Sample {sample}, MSE {mse:.7f}, MAE {mae:.4f}')
    plt.tight_layout()
    fig.subplots_adjust(top=0.975)
    plt.show()