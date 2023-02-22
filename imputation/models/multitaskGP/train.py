import argparse
import copy
import warnings
import random
import os

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as f

from model import HadamardGP, HadamardGaussianLikelihood
from toy_dataset import data_utils

warnings.filterwarnings('ignore', '.*dataloader.*does not have many workers.*', )
warnings.filterwarnings('ignore', '.*triangular_solve is deprecated.*', )

# mae/ mse for both train/ evaluation process
# mae/mse/mll for both input/ eval/ gt (9 pieces of information) -> for both training and eval process (18 pieces of information)
# with torch.no_grad() -> mae/mse mll(gt...)
# best model definition: based mll

def mask_collate(batch):
    #
    masks = batch[:, :, 4: -1]
    values = batch[:, :, :4]
    t = batch[0, :, -1]

    # further create input mask and eval mask
    return masks, values, t


def sort_by_task_t(task_tensor, t_tensor, x_tensor):
    task_arr = task_tensor.cpu().detach().numpy()
    t_arr = t_tensor.cpu().detach().numpy()
    x_arr = x_tensor.cpu().detach().numpy()
    
    concat_df = pd.DataFrame({'task': task_arr,
                              't': t_arr,
                              'x': x_arr})
    concat_df_sorted = concat_df.sort_values(['task', 't'], ascending=[True, True])
    task_tensor_sorted = torch.tensor(concat_df_sorted['task'].to_numpy()).type_as(task_tensor)
    t_tensor_sorted = torch.tensor(concat_df_sorted['t'].to_numpy()).type_as(t_tensor)
    x_tensor_sorted = torch.tensor(concat_df_sorted['x'].to_numpy()).type_as(x_tensor)
    
    return task_tensor_sorted, t_tensor_sorted, x_tensor_sorted

class MGPImputer(pl.LightningModule):
    def __init__(self, pars=None) -> None:
        super().__init__()

        # self.pars = pars
        # self.data_ready = False

        # Model parts
        # num_tasks = len(pars['dyn_vars'])
        self.data_mode = "simulation"
        self.mae_loss = torch.nn.L1Loss()
        # for plotting figures
        self.task_names = ['Noise', 'Trend', "Seasonality", "Trend + Seasonality"]
        self.num_tasks = 4
        # TODO: perhaps shouldn't be a fixed value, and should be varied during training
        # pick the eval that works best at a fixed sample_tp (as prior knowledge of missing-rate for the dataset)
        self.sample_tp = 0.4
        # randomly pick a missing rate in the interval for model's robustness
        self.sample_tp_interval = [0.1, 0.8]
        self.mgp = HadamardGP(
            train_inputs = None,
            train_targets = None,
            likelihood=HadamardGaussianLikelihood(num_tasks=4),
            num_tasks=4,
            num_kernels=10,
            rank=4
        )
        self.mll = ExactMarginalLogLikelihood(self.mgp.likelihood, self.mgp)

    def forward(self, x):
        return self.mgp(x)

    def _calculate_loss(self, batch):
        recon_loss_target = 0
        recon_loss_context = 0
        recon_loss_gt = 0
        
        mae_loss_target = 0
        mae_loss_context = 0
        mae_loss_gt = 0
        
        mse_loss_target = 0
        mse_loss_context = 0
        mse_loss_gt = 0

        num_seq = 0
        if self.data_mode == "simulation":
            batch, gt_batch = batch
            masks, values, t = mask_collate(batch)
            _, gt_values, _ = mask_collate(gt_batch)

        else:
            masks, values, t = mask_collate(batch)
        for mask, value in zip(masks, values):  # Static features and sepsis label not needed
            # non-vectorized version
            # go over each task within each sequence
            # tasks_context = []
            # ts_context = []
            # xs_context = []
            #
            # tasks_target = []
            # ts_target = []
            # xs_target = []
            # for i in range(4):
            #     value_i = value[:, i]
            #     mask_i = mask[:, i]
            #
            #     artificial_missingness = 0.4
            #     mask_i_input = torch.zeros_like(mask_i)
            #     mask_i_eval = torch.zeros_like(mask_i)
            #     for entry_idx, mask_entry in enumerate(mask_i):
            #         if mask_entry != 0:
            #             if random.uniform(0, 1) >= artificial_missingness:
            #                 mask_i_input[entry_idx] = 1
            #             else:
            #                 mask_i_eval[entry_idx] = 1
            #     mask_i_input = mask_i_input.type_as(mask_i)
            #     mask_i_eval = mask_i_eval.type_as(mask_i)
            #
            #     ts_context.append(t[mask_i_input == 1])
            #     xs_context.append(value_i[mask_i_input == 1])
            #     task_context = torch.full((ts_context[-1].shape[0], 1), dtype=torch.long, fill_value=i).type_as(t)
            #     tasks_context.append(task_context)
            #
            #     ts_target.append(t[mask_i_eval == 1])
            #     xs_target.append(value_i[mask_i_eval == 1])
            #     task_target = torch.full((ts_target[-1].shape[0], 1), dtype=torch.long, fill_value=i).type_as(t)
            #     tasks_target.append(task_target)
            #
            # full_t_context = torch.cat(ts_context)
            # full_tasks_context = torch.cat(tasks_context)
            # full_x_context = torch.cat(xs_context)
            #
            # full_t_target = torch.cat(ts_target)
            # full_tasks_target = torch.cat(tasks_target)
            # full_x_target = torch.cat(xs_target)
            #
            # input_task_t_context = torch.cat((full_tasks_context, full_t_context.view(full_t_context.shape[0], 1)), dim=-1)
            #
            # input_task_t_target = torch.cat((full_tasks_target, full_t_target.view(full_t_target.shape[0], 1)), dim=-1)



            # vectorized version
            ts = t.view(-1, 1).repeat(1, self.num_tasks)
            tasks = torch.tensor([task_i for task_i in range(self.num_tasks)]).view(1, -1).repeat(t.shape[0], 1).type_as(ts)

            # prevent input and eval arrays from sharing the same memory
            # generate mask tensor
            # self.sample_tp
            if self.train_mode:
                artificial_missingness = random.uniform(self.sample_tp_interval[0], self.sample_tp_interval[-1])
            else:
                artificial_missingness = self.sample_tp

            mask_input = copy.deepcopy(mask)
            mask_eval = copy.deepcopy(mask)
            mask_input[mask == 1] = torch.from_numpy(np.random.choice([0, 1],
                                                                      mask_input[mask == 1].shape[0],
                                                                      p=[artificial_missingness,
                                                                         1 - artificial_missingness])).type_as(mask)
            mask_eval[mask == 1] = 1 - mask_input[mask == 1]

            # fetch input data using boolean indexing
            full_t_context = ts[mask_input == 1]
            full_tasks_context = tasks[mask_input == 1]
            full_x_context = value[mask_input == 1]
            # full_tasks_context, full_t_context, full_x_context = sort_by_task_t(full_tasks_context, full_t_context, full_x_context)
            input_task_t_context = torch.cat(
                (full_tasks_context.view(full_tasks_context.shape[0], 1),
                 full_t_context.view(full_t_context.shape[0], 1)),
                dim=-1)

            # fetch eval data using boolean indexing
            full_t_target = ts[mask_eval == 1]
            full_tasks_target = tasks[mask_eval == 1]
            full_x_target = value[mask_eval == 1]
            # full_tasks_target, full_t_target, full_x_target = sort_by_task_t(full_tasks_target, full_t_target, full_x_target)
            input_task_t_target = torch.cat(
                 (full_tasks_target.view(full_tasks_target.shape[0], 1),
                  full_t_target.view(full_t_target.shape[0], 1)),
                 dim=-1)
            
            if self.data_mode == "simulation":
                full_t_gt = ts[mask != 1]
                full_tasks_gt = tasks[mask != 1]
                full_x_gt = gt_values[num_seq][mask != 1]
                # full_tasks_gt, full_t_gt, full_x_gt = sort_by_task_t(full_tasks_gt, full_t_gt, full_x_gt)
                input_task_t_gt = torch.cat(
                    (full_tasks_gt.view(full_tasks_gt.shape[0], 1),
                     full_t_gt.view(full_t_gt.shape[0], 1)),
                    dim=-1)

            # contextualize the model
            # form loss for both context points and target points to stabilize training process

            if self.train_mode:
                input_task_t_target = torch.cat((input_task_t_context, input_task_t_target), dim=0)
                full_x_target = torch.cat((full_x_context, full_x_target), dim=0)
                self.mgp.set_train_data(input_task_t_target, full_x_target, strict=False)
            else:
                self.mgp.set_train_data(input_task_t_context, full_x_context, strict=False)

            output = self(input_task_t_target)
            recon_loss_target += -self.mll(output, full_x_target, [input_task_t_target])

            # in order not to slow down training process (only the evaluation process will be slowed down)
            if not self.train_mode:
                with gpytorch.settings.debug(False), gpytorch.settings.fast_pred_var(), torch.no_grad():
                    self.mgp.eval()
                    # calculate losses for input
                    observed_pred = self.mgp.likelihood(self.mgp(input_task_t_target), [input_task_t_target])
                    mae_loss_target += self.mae_loss(observed_pred.mean, full_x_target)
                    mse_loss_target += f.mse_loss(observed_pred.mean, full_x_target)

                    # calculate losses for context
                    observed_pred = self.mgp.likelihood(self.mgp(input_task_t_context), [input_task_t_context])
                    mae_loss_context += self.mae_loss(observed_pred.mean, full_x_context)
                    mse_loss_context += f.mse_loss(observed_pred.mean, full_x_context)

                    output = self(input_task_t_context)
                    recon_loss_context += -self.mll(output, full_x_context, [input_task_t_context])

                    # calculate losses for ground truth
                    if self.data_mode == "simulation":
                        observed_pred = self.mgp.likelihood(self.mgp(input_task_t_gt), [input_task_t_gt])
                        mae_loss_gt += self.mae_loss(observed_pred.mean, full_x_gt)
                        mse_loss_gt += f.mse_loss(observed_pred.mean, full_x_gt)

                        output = self(input_task_t_gt)
                        recon_loss_gt += -self.mll(output, full_x_gt, [input_task_t_gt])


            num_seq += 1
        return recon_loss_target/num_seq, recon_loss_context/num_seq, recon_loss_gt/num_seq, \
            mae_loss_target/num_seq, mae_loss_context/num_seq, mae_loss_gt/num_seq, \
            mse_loss_target/num_seq, mse_loss_context/num_seq, mse_loss_gt/num_seq

    def training_step(self, batch, batch_idx):
        self.train_mode = True
        recon_loss_target, recon_loss_context, recon_loss_gt, \
            mae_loss_target, mae_loss_context, mae_loss_gt, \
            mse_loss_target, mse_loss_context, mse_loss_gt = self._calculate_loss(batch)
        # self.log("t_recon_loss", recon_loss)
        # self.log("t_mae_loss", mae_loss)
        # self.log("t_mse_loss", mse_loss)
        if self.data_mode == "simulation":
            self.log_dict({"t_recon_loss_target": recon_loss_target, "t_recon_loss_context": recon_loss_context, "t_recon_loss_gt": recon_loss_gt,
                           "t_mae_loss_target": mae_loss_target, "t_mae_loss_context": mae_loss_context, "t_mae_loss_gt": mae_loss_gt,
                           "t_mse_loss_target": mse_loss_target, "t_mse_loss_context": mse_loss_context, "t_mse_loss_gt": mse_loss_gt}, on_step=True)
        else:
            self.log_dict({"t_recon_loss_target": recon_loss_target, "t_recon_loss_context": recon_loss_context,
                           "t_mae_loss_target": mae_loss_target, "t_mae_loss_context": mae_loss_context,
                           "t_mse_loss_target": mse_loss_target, "t_mse_loss_context": mse_loss_context}, on_step=True)
        # return {"t_recon_loss": recon_loss, "t_mae_loss": mae_loss, "t_mse_loss": mse_loss}
        # tensorboard_logs = {"t_recon_loss": recon_loss, "t_mae_loss": mae_loss, "t_mse_loss": mse_loss}
        # return {"loss": recon_loss, 'log': tensorboard_logs}
        return recon_loss_target

    def training_epoch_end(self, outputs):
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        self.train_mode = False
        self.train()  # necessary because gpytorch behaves differently in eval
        with gpytorch.settings.debug(False), torch.no_grad():
            recon_loss_target, recon_loss_context, recon_loss_gt, \
                mae_loss_target, mae_loss_context, mae_loss_gt, \
                mse_loss_target, mse_loss_context, mse_loss_gt = self._calculate_loss(batch)
        self.eval()
        # self.log("v_recon_loss", recon_loss)
        # self.log("v_mae_loss", mae_loss)
        # self.log("v_mse_loss", mse_loss)
        if self.data_mode == "simulation":
            self.log_dict({"v_recon_loss_target": recon_loss_target, "v_recon_loss_context": recon_loss_context, "v_recon_loss_gt": recon_loss_gt,
                           "v_mae_loss_target": mae_loss_target, "v_mae_loss_context": mae_loss_context, "v_mae_loss_gt": mae_loss_gt,
                           "v_mse_loss_target": mse_loss_target, "v_mse_loss_context": mse_loss_context, "v_mse_loss_gt": mse_loss_gt}, on_step=True)
        else:
            self.log_dict({"v_recon_loss_target": recon_loss_target, "v_recon_loss_context": recon_loss_context,
                           "v_mae_loss_target": mae_loss_target, "v_mae_loss_context": mae_loss_context,
                           "v_mse_loss_target": mse_loss_target, "v_mse_loss_context": mse_loss_context}, on_step=True)
        # return {"v_recon_loss": recon_loss, "v_mae_loss": mae_loss, "v_mse_loss": mse_loss}
        # tensorboard_logs = {"v_recon_loss": recon_loss, "v_mae_loss": mae_loss, "v_mse_loss": mse_loss}
        # return {"v_loss": recon_loss, 'log': tensorboard_logs}
        return recon_loss_target


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def init_kernels(self):
        pass

    def prepare_data(self, force=False) -> None:
        name = 'toydataset_50000'
        path = data_utils.datasets_dict[name]
        dataset = data_utils.ToyDataDf(path)
        dataset.create_mcar_missingness(0.6, -1)
        dataloader_dict = dataset.prepare_data_mtan(batch_size=128)
        self.dataloader_dict = dataloader_dict

    def train_dataloader(self):
        if self.data_mode == "simulation":
            return zip(self.dataloader_dict['train'], self.dataloader_dict['train_ground_truth'])
        else:
            return self.dataloader_dict['train']

    def val_dataloader(self):
        if self.data_mode == "simulation":
            return zip(self.dataloader_dict['validation'], self.dataloader_dict['validation_ground_truth'])
        else:
            return self.dataloader_dict['validation']

    def test_dataloader(self):
        pass

    def training_epoch_end(self, outputs):
        pass
    def validation_epoch_end(self, outputs):
        pass

    # other customized class method
    # sample_tp can vary during inference
    def impute(self, mask, value, t, sample_tp):
        # if there should be no artificial missingness, sample_tp should be set to 0

        # vectorize ts and tasks tensor for masking
        ts = t.view(-1, 1).repeat(1, self.num_tasks)
        tasks = torch.tensor([task_i for task_i in range(self.num_tasks)]).view(1, -1).repeat(t.shape[0], 1)

        # vectorized implementation
        # prevent input and eval arrays from sharing the same memory
        artificial_missingness = sample_tp
        mask_input = copy.deepcopy(mask)
        mask_eval = copy.deepcopy(mask)
        mask_input[mask == 1] = torch.from_numpy(np.random.choice([0, 1],
                                                                  mask_input[mask == 1].shape[0],
                                                                  p=[artificial_missingness,
                                                                     1 - artificial_missingness])).type_as(mask)
        mask_eval[mask == 1] = 1 - mask_input[mask == 1]

        full_t_context = ts[mask_input == 1]
        full_tasks_context = tasks[mask_input == 1]
        full_x_context = value[mask_input == 1]

        input_task_t_context = torch.cat(
            (full_tasks_context.view(full_tasks_context.shape[0], 1),
             full_t_context.view(full_t_context.shape[0], 1)),
            dim=-1)
        self.mgp.set_train_data(input_task_t_context, full_x_context, strict=False)

        # vectorized version
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            input_task_t_target = torch.cat((tasks.view(-1, 1), ts.view(-1, 1)), dim=-1)
            self.mgp.eval()
            observed_pred = self.mgp.likelihood(self.mgp(input_task_t_target), [input_task_t_target])
            lower_2d, upper_2d = observed_pred.confidence_region()
            std = (upper_2d - observed_pred.mean) / 2.0
            upper_1d = observed_pred.mean + std
            lower_1d = observed_pred.mean - std

        quantile_high = upper_1d.view(-1, self.num_tasks)
        quantile_low = lower_1d.view(-1, self.num_tasks)
        pred_mean = observed_pred.mean.view(-1, self.num_tasks)

        mask_dict = {'input': mask_input, 'eval': mask_eval, 'gt': mask}

        self.mask_dict = mask_dict
        self.pred_mean = pred_mean
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.observed_pred = observed_pred
        return pred_mean, quantile_low, quantile_high, mask_dict, observed_pred

    # @staticmethod
    def sample_seq(self):
        # vectorized version
        return self.observed_pred.sample().view(-1, self.num_tasks)

    def plot_seq(self, t, value,
                 plot_idx=0,
                 plot_save_folder=None,
                 gt_value=None,
                 sampled_seqs=None):
        f, axes = plt.subplots(self.num_tasks, 1, figsize=(10, 28))

        for i in range(self.num_tasks):
            axes[i].plot(t[self.mask_dict['input'][:, i] == 1].detach().numpy(),
                         value[:, i][self.mask_dict['input'][:, i] == 1].detach().numpy(), 'k*')
            axes[i].plot(t[self.mask_dict['gt'][:, i] == 1].detach().numpy(),
                         value[:, i][self.mask_dict['gt'][:, i] == 1].detach().numpy(), 'yo', alpha=0.4)
            # Predictive mean as blue line
            axes[i].plot(t.detach().numpy(), self.pred_mean[:, i].detach().numpy(), 'b')
            # Shade in confidence
            axes[i].fill_between(t.detach().numpy(),
                                 self.quantile_low[:, i].detach().numpy(),
                                 self.quantile_high[:, i].detach().numpy(), alpha=0.5)
            # if we also want to visualize ground truth
            if gt_value is not None:
                axes[i].plot(t.detach().numpy(), gt_value[:, i].detach().numpy(), 'r')
                axes[i].legend(['Input Data', 'Input + Eval Data', 'Mean', 'Confidence', 'ground truth'])
            else:
                axes[i].legend(['Input Data', 'Input + Eval Data', 'Mean', 'Confidence'])

            if sampled_seqs is not None:
                for sampled_seq in sampled_seqs:
                    axes[i].plot(t.detach().numpy(), sampled_seq[:, i].detach().numpy(), 'g', alpha=0.1)

            axes[i].set_title(self.task_names[i])
        save_path = os.path.join(plot_save_folder, f'{plot_idx}th_plot.png')
        f.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    mod = MGPImputer()
    mod.prepare_data()
    num_train_batches = len(mod.dataloader_dict['train'])

    model_weights_folder_path = "./model_weights"
    os.makedirs(model_weights_folder_path, exist_ok=True)

    # define the model saving callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='v_recon_loss_target',  # the metric to monitor
        dirpath=model_weights_folder_path,
        filename='best_model-{v_recon_loss_target:.2f}',  # the file name pattern for the saved checkpoint
        save_top_k=1,  # save the top 1 model
        mode='min'  # the mode to compare the monitored metric (either 'min' or 'max')
    )

    # define the TensorBoard logger
    # logger = pl.loggers.TensorBoardLogger("./logs", name="multitask_gp")

    # define the TensorBoard callback
    # tb_callback = pl.callbacks.TensorBoardCallback(
    #     log_every_n_steps=10,
    #     write_graph=True
    # )

    # now set it to None, since gpu doesn't accelerate the training process
    trainer = pl.Trainer(max_epochs=3, gpus=1,
                         # log_every_n_steps=num_train_batches,
                         # logger=logger,
                         callbacks=[checkpoint_callback],
                         fast_dev_run=False)
    trainer.fit(mod)
