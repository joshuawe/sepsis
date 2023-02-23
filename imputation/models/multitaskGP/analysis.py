import os

from toy_dataset import data_utils
from train import MGPImputer, mask_collate

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
    best_model_path = "./model_weights/best_model-v_recon_loss_target=1.11-epoch=0.ckpt"
    imputer = MGPImputer.load_from_checkpoint(best_model_path)

    # configure plot saving path and number of plots to be saved
    plot_save_folder = "./test_plots"
    os.makedirs(plot_save_folder, exist_ok=True)
    num_sampled_seqs_per_plot = 10
    max_num_plots = 30

    num_plots = 0
    gt_iter = iter(gt_data_loader)
    for i_batch, sample in enumerate(data_loader):
        gt_sample = next(gt_iter)
        _, gt_values, _ = mask_collate(gt_sample)

        masks, values, t = mask_collate(sample)
        for mask, value, gt_value in zip(masks, values, gt_values):  # Static features and sepsis label not needed
            pred_mean, quantile_low, quantile_high, mask_dict, observed_pred = imputer.impute(mask, value, t, sample_tp=0.0)
            sampled_seqs = []

            for i in range(num_sampled_seqs_per_plot):
                sampled_seq = imputer.sample_seq()
                sampled_seqs.append(sampled_seq)

            if num_plots < max_num_plots:
                imputer.plot_seq(t, value,
                                 plot_idx=num_plots,
                                 plot_save_folder=plot_save_folder,
                                 gt_value=gt_value,
                                 sampled_seqs=sampled_seqs)
                num_plots += 1
                print(f"length of sampled_seqs {len(sampled_seqs)}")
            else:
                break
