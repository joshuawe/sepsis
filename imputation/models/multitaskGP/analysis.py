import os
import time
import torch
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
    target_ts = torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5]) # [50]
    target_t = target_ts

    # visualisation for debugging the model
    if VISUALISE_FOR_DEBUG:
        for i_batch, sample in enumerate(data_loader):
            gt_sample = next(gt_iter)
            _, gt_values, _ = mask_collate(gt_sample)

            masks, values, ts = mask_collate(sample)
            time_start_batch = time.time()
            for mask, value, gt_value, t in zip(masks, values, gt_values, ts):  # Static features and sepsis label not needed

                pred_mean_target, quantile_low_target, \
                    quantile_high_target, mask_dict, _ = \
                    imputer.impute(mask, value, t, sample_tp=0.2, target_t=target_t)

                sampled_seqs_target = imputer.sample_seq(sample_size=num_sampled_seqs_per_plot)

                sampled_seqs = imputer.sample_seq(sample_size=num_sampled_seqs_per_plot)

                if num_plots < max_num_plots:
                    imputer.plot_seq(t, value,
                                     # sampled_seqs_target_t=sampled_seqs_target,
                                     target_t=target_t,
                                     plot_idx=num_plots,
                                     plot_save_folder=plot_save_folder,
                                     gt_value=gt_value,
                                     sampled_seqs=sampled_seqs)
                    num_plots += 1
                    print(f"length of sampled_seqs {sampled_seqs.shape[0]}")
                else:
                    break
