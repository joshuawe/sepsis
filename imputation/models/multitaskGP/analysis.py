import os

from train import MGPImputer, mask_collate
from toy_dataset import data_utils
import torch
import random
import gpytorch
import matplotlib.pyplot as plt
import copy
import numpy as np

# 50 * 4
def impute(imputer, mask, value, t, sample_tp, num_tasks=4):
    # go over each task within each sequence
    tasks_context = []
    ts_context = []
    xs_context = []
    
    # vectorize ts and tasks tensor for masking
    ts = t.view(-1, 1).repeat(1, num_tasks)
    tasks = torch.tensor([task_i for task_i in range(num_tasks)]).view(1, -1).repeat(t.shape[0], 1)

    # vectorized implementation
    # prevent input and eval arrays from sharing the same memory
    artificial_missingness = sample_tp
    mask_input = copy.deepcopy(mask)
    mask_eval = copy.deepcopy(mask)
    mask_input[mask==1] = torch.from_numpy(np.random.choice([0, 1],
                                                            mask_input[mask==1].shape[0],
                                                            p=[artificial_missingness, 1-artificial_missingness])).type_as(mask)
    mask_eval[mask==1] = 1-mask_input[mask==1]

    full_t_context = ts[mask_input==1]
    full_tasks_context = tasks[mask_input==1]
    full_x_context = value[mask_input==1]

    # non-vectorized implementation
    # mask_input = []
    # mask_eval = []
    #
    # for i in range(num_tasks):
    #     value_i = value[:, i]
    #     mask_i = mask[:, i]
    #
    #     artificial_missingness = sample_tp
    #     mask_i_input = torch.zeros_like(mask_i)
    #     mask_i_eval = torch.zeros_like(mask_i)
    #     # randomly generate mask for input/ eval from the observed entries
    #     for entry_idx, mask_entry in enumerate(mask_i):
    #         if mask_entry != 0:
    #             if random.uniform(0, 1) >= artificial_missingness:
    #                 mask_i_input[entry_idx] = 1
    #             else:
    #                 mask_i_eval[entry_idx] = 1
    #     mask_i_input = mask_i_input.type_as(mask_i)
    #     mask_i_eval = mask_i_eval.type_as(mask_i)
    #
    #     # fetch input data
    #     ts_context.append(t[mask_i_input == 1])
    #     xs_context.append(value_i[mask_i_input == 1])
    #     task_context = torch.full((ts_context[-1].shape[0], 1), dtype=torch.long, fill_value=i).type_as(t)
    #     tasks_context.append(task_context)
    #
    #     mask_input.append(mask_i_input.unsqueeze(-1))
    #     mask_eval.append(mask_i_eval.unsqueeze(-1))
    #
    # # condition the MGP on full context data
    # full_t_context = torch.cat(ts_context)
    # full_tasks_context = torch.cat(tasks_context)
    # full_x_context = torch.cat(xs_context)
    # target_x_context = full_x_context
    input_task_t_context = torch.cat(
        (full_tasks_context.view(full_tasks_context.shape[0], 1),
         full_t_context.view(full_t_context.shape[0], 1)),
        dim=-1)
    imputer.mgp.set_train_data(input_task_t_context, full_x_context, strict=False)

    # # get full masks
    # full_mask_input = torch.cat(mask_input, dim=-1)
    # full_mask_eval = torch.cat(mask_eval, dim=-1)

    # get mean prediction and CI over the whole seq
    # non-vectorized version
    # quantile_low = []
    # quantile_high = []
    # pred_mean = []
    # pred_dist_dict = {}
    #
    # for i in range(num_tasks):
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         # output_val = model(input_task_t)
    #
    #         task_target = torch.full((t.shape[0], 1), dtype=torch.long, fill_value=i)
    #         input_task_t_target = torch.cat((task_target, t.view(t.shape[0], 1)), dim=-1)
    #
    #         imputer.mgp.eval()  # necessary if you want to plot lower and upper bound!!!
    #         observed_pred = imputer.mgp.likelihood(imputer.mgp(input_task_t_target), [input_task_t_target])
    #         lower_2d, upper_2d = observed_pred.confidence_region()
    #         std = (upper_2d - observed_pred.mean) / 2.0
    #         upper_1d = observed_pred.mean + std
    #         lower_1d = observed_pred.mean - std
    #         quantile_high.append(upper_1d.unsqueeze(-1))
    #         quantile_low.append(lower_1d.unsqueeze(-1))
    #         pred_mean.append(observed_pred.mean.unsqueeze(-1))
    #         pred_dist_dict[f'channel_{i}'] = observed_pred
    # quantile_high = torch.cat(quantile_high, dim=-1)
    # quantile_low = torch.cat(quantile_low, dim=-1)
    # pred_mean = torch.cat(pred_mean, dim=-1)

    # vectorized version
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        input_task_t_target = torch.cat((tasks.view(-1, 1), ts.view(-1, 1)), dim=-1)
        imputer.mgp.eval()
        observed_pred = imputer.mgp.likelihood(imputer.mgp(input_task_t_target), [input_task_t_target])
        lower_2d, upper_2d = observed_pred.confidence_region()
        std = (upper_2d - observed_pred.mean) / 2.0
        upper_1d = observed_pred.mean + std
        lower_1d = observed_pred.mean - std

    quantile_high = upper_1d.view(-1, num_tasks)
    quantile_low = lower_1d.view(-1, num_tasks)
    pred_mean = observed_pred.mean.view(-1, num_tasks)


    mask_dict = {'input': mask_input, 'eval': mask_eval, 'gt': mask}
    return pred_mean, quantile_low, quantile_high, mask_dict, observed_pred


def sample_seq(pred_distribution_dict, num_tasks=4):
    # non-vectorized version
    # sampled_seq = []
    # for i in range(num_tasks):
    #     sampled_seq.append(pred_distribution_dict[f'channel_{i}'].sample().unsqueeze(-1))
    # return torch.cat(sampled_seq, dim=-1)

    # vectorized version
    return pred_distribution_dict.sample().view(-1, num_tasks)



def plot_seq(pred_mean, quantile_low, quantile_high, mask_dict,
             gt_value, t, value,
             num_tasks=4,
             plot_idx=0,
             plot_save_folder=None):
    f, axes = plt.subplots(num_tasks, 1, figsize=(10, 28))
    titles = ['Noise', 'Trend', "Seasonality", "Trend + Seasonality"]
    for i in range(num_tasks):
        axes[i].plot(t[mask_dict['input'][:, i] == 1].detach().numpy(),
                     value[:, i][mask_dict['input'][:, i] == 1].detach().numpy(), 'k*')
        axes[i].plot(t[mask_dict['gt'][:, i] == 1].detach().numpy(),
                     value[:, i][mask_dict['gt'][:, i] == 1].detach().numpy(), 'yo', alpha=0.4)
        # Predictive mean as blue line
        axes[i].plot(t.detach().numpy(), pred_mean[:,i].detach().numpy(), 'b')
        # Shade in confidence
        axes[i].fill_between(t.detach().numpy(),
                             quantile_low[:,i].detach().numpy(),
                             quantile_high[:,i].detach().numpy(), alpha=0.5)

        axes[i].plot(t.detach().numpy(), gt_value[:,i].detach().numpy(), 'r')
        # ax.set_ylim([-3, 3])
        # axes[i].plot(t_val.detach().numpy(), gt_values_val[0, :, i].detach().numpy(), 'r')
        axes[i].legend(['Input Data', 'Input + Eval Data', 'Mean', 'Confidence',
                        'ground truth'
                        ])
        axes[i].set_title(titles[i])
    save_path = os.path.join(plot_save_folder, f'{plot_idx}th_plot.png')
    f.savefig(save_path)


if __name__ == "__main__":
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

    best_model_path = "./model_weights/best_model-v_recon_loss_target=-2.01.ckpt"
    imputer = MGPImputer.load_from_checkpoint(best_model_path)

    data_loader = val_loader
    gt_data_loader = gt_validation_loader

    plot_save_folder = "./test_plots"
    os.makedirs(plot_save_folder, exist_ok=True)

    num_plots = 0
    max_num_plots = 30
    gt_iter = iter(gt_data_loader)
    for i_batch, sample in enumerate(data_loader):
        gt_sample = next(gt_iter)
        _, gt_values, _ = mask_collate(gt_sample)

        masks, values, t = mask_collate(sample)
        for mask, value, gt_value in zip(masks, values, gt_values):  # Static features and sepsis label not needed
            # new
            pred_mean, quantile_low, quantile_high, mask_dict, observed_pred = imputer.impute(mask, value, t, sample_tp=0.0)

            sampled_seqs = []
            for i in range(10):
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


            # legacy
            # pred_mean, quantile_low, quantile_high, mask_dict, pred_dist_dict = impute(imputer, mask, value, t, sample_tp=0.4)
            # sampled_seq = sample_seq(pred_dist_dict)
            # if num_plots < max_num_plots:
            #     plot_seq(pred_mean, quantile_low, quantile_high, mask_dict, gt_value, t, value,
            #              plot_save_folder=plot_save_folder, plot_idx=num_plots)
            #     num_plots += 1
            #
            #     print("ok!")
            # else:
            #     break




