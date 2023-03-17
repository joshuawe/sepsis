import os
import time

from toy_dataset import data_utils
from train import MGPImputer, mask_collate

VISUALISE_FOR_DEBUG = False

if __name__ == "__main__":
    # configure synthetic dataset
    name = 'toydataset_50000'
    path = data_utils.datasets_dict[name]
    dataset = data_utils.ToyDataDf(path)
    dataset.create_mcar_missingness(0.6, -1)
    dataloader_dict = dataset.prepare_data_mtan(batch_size=128)
    train_loader = dataloader_dict['train']
    gt_train_loader = dataloader_dict['train_ground_truth']
    val_loader = dataloader_dict['validation']
    gt_validation_loader = dataloader_dict['validation_ground_truth']
    test_loader = val_loader

    # can be changed to train_loader and gt_train_loader respectively
    data_loader = val_loader
    gt_data_loader = gt_validation_loader

    # configure model
    best_model_path = "./model_weights/best_model-v_recon_loss_target=-1.48-epoch=2.ckpt"
    imputer = MGPImputer.load_from_checkpoint(best_model_path)

    # configure plot saving path and number of plots to be saved
    if VISUALISE_FOR_DEBUG:
        plot_save_folder = "./test_plots"
        os.makedirs(plot_save_folder, exist_ok=True)
        num_sampled_seqs_per_plot = 1000
        max_num_plots = 30

        num_plots = 0
        gt_iter = iter(gt_data_loader)

    # for showing how to do batch-wise imputation and sampling + speed benchmarking
    time_start_total = time.time()
    for i_batch, sample in enumerate(data_loader):
        time_start_batch = time.time()
        masks, values, t = mask_collate(sample)
        pred_mean_batch, quantile_low_batch, quantile_high_batch, sampled_seqs_batch, mask_dict_batch = \
            imputer.impute_and_sample_in_batch(masks, values, t, sample_tp=0.2, sample_size=1000)
        time_delta_batch = time.time() - time_start_batch
        print(f"this batch takes {time_delta_batch} for processing")
    time_delta_total = time.time() - time_start_total
    print(f"all batches take {time_delta_total} for processing")

    # visualisation for debugging the model
    if VISUALISE_FOR_DEBUG:
        for i_batch, sample in enumerate(data_loader):
            gt_sample = next(gt_iter)
            _, gt_values, _ = mask_collate(gt_sample)

            masks, values, t = mask_collate(sample)
            time_start_batch = time.time()
            for mask, value, gt_value in zip(masks, values, gt_values):  # Static features and sepsis label not needed
                pred_mean, quantile_low, quantile_high, mask_dict, observed_pred = imputer.impute(mask, value, t, sample_tp=0.0)
                sampled_seqs = imputer.sample_seq(sample_size=num_sampled_seqs_per_plot)

                if num_plots < max_num_plots:
                    imputer.plot_seq(t, value,
                                     plot_idx=num_plots,
                                     plot_save_folder=plot_save_folder,
                                     gt_value=gt_value,
                                     sampled_seqs=sampled_seqs)
                    num_plots += 1
                    print(f"length of sampled_seqs {sampled_seqs.shape[0]}")
                else:
                    break
