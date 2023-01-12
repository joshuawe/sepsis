# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from random import SystemRandom
import models
import utils

# parser = argparse.ArgumentParser()
# parser.add_argument('--niters', type=int, default=2000, help='Training epochs or iterations.')
# parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
# parser.add_argument('--latent-dim', type=int, default=32, help='Dimension of latent states. The latent representation consists of a mean and a logvar for each latent dimension.')
# parser.add_argument('--rec-hidden', type=int, default=32, help='???')
# parser.add_argument('--width', type=int, default=50, help='???')
# parser.add_argument('--embed-time', type=int, default=128, help='???')
# parser.add_argument('--num-ref-points', type=int, default=128, help='???')
# parser.add_argument('--k-iwae', type=int, default=10, help='??? Number of samples drawn from latent space.')
# parser.add_argument('--save', action='store_true', help='Flag for saving trained model.')
# parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
# parser.add_argument('--n', type=int, default=8000, help='Number of data points to be created. Only for synthetic dataset.')
# parser.add_argument('--batch-size', type=int, default=50, help='Batch size.')
# parser.add_argument('--norm', action='store_true', help='???')
# parser.add_argument('--kl-annealing', action='store_true', help='Annealing (=scheduling) for KL weighting.')
# parser.add_argument('--kl-zero', action='store_true', help='If set, the KL weighting is zero in the loss function.')
# parser.add_argument('--enc-num-heads', type=int, default=1, help='???')
# parser.add_argument('--dataset', type=str, default='toy', help='Name of Dataset to be used.')
# parser.add_argument('--dropout', type=float, default=0.0, help='Dropout regularization value between [0,1) .')
# parser.add_argument('--intensity', action='store_true', help='???')
# parser.add_argument('--net', type=str, default='hetvae', help='Which deep learning model to use.')
# parser.add_argument('--const-var', action='store_true', help='???')
# parser.add_argument('--var-per-dim', action='store_true', help='???')
# parser.add_argument('--std', type=float, default=0.1, help='???')
# parser.add_argument('--sample-tp', type=float, default=0.5, help='How many timepoints to be subsampled. Fraction value between (0,1] .')
# parser.add_argument('--bound-variance', action='store_true', help='???')
# parser.add_argument('--shuffle', action='store_true', help='??? Something with shuffling time points.')
# parser.add_argument('--recon-loss', action='store_true', help='??? Something with sub-sampling time points.')
# parser.add_argument('--normalize-input', type=str, default='znorm', help='???')
# parser.add_argument('--mse-weight', type=float, default=0.0, help='???')
# parser.add_argument('--elbo-weight', type=float, default=1.0, help='???')
# parser.add_argument('--mixing', type=str, default='concat', help='How the deterministic and probabilistic paths of encoder are used. Options are concat_and_mix, concat, seperate, interp_only and na.')
# args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 10000000)
    experiment_id = 'hetvae_' + str(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    log_dir = '~/Documents/sepsis/imputation/runs/' + experiment_id + '/'
    print(args, '\nexperiment_id:', experiment_id, '\nlog_dir:', log_dir)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'toy':
        # data_obj = utils.get_synthetic_data(args)
        data_obj = utils.get_toydata(args.batch_size)
    elif args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args.batch_size)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data(args.batch_size, filter_anomalies=True)

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    union_tp = utils.union_time(train_loader)

    net = models.load_network(args, dim, union_tp)
    params = list(net.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    print('parameters:', utils.count_parameters(net))

    # Tensorboard Summary Writer
    writer = SummaryWriter(log_dir=log_dir)
    # with two plots in one
    layout = {"Training vs. Validation": {
                "loss": ["Multiline", ["train_loss", "val_loss"]],
                "mae": ["Multiline", ["mae_train", "mae_val"]],
                "mse": ["Multiline", ["mse_train", "mse_val"]],
                "avg_kl": ["Multiline", ["avg_kl_train", "avg_kl_val"]],
                "avg_kl": ["Multiline", ["-avg_loglik_train", "-avg_loglik_val"]], 
                },
    }
    writer.add_custom_scalars(layout)

    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_loglik, avg_kl, mse, mae = 0, 0, 0, 0
        if args.kl_annealing:
            wait_until_kl_inc = 10000
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.999 ** (itr - wait_until_kl_inc))
        elif args.kl_zero:
            kl_coef = 0
        else:
            kl_coef = 1
        for train_batch in train_loader:
            batch_len = train_batch.shape[0]
            train_batch = train_batch.to(device)
            if args.dataset == 'toy':
                subsampled_mask = torch.zeros_like(
                    train_batch[:, :, dim:2 * dim]).to(device)
                seqlen = train_batch.size(1)
                for i in range(batch_len):
                    length = np.random.randint(low=3, high=10)
                    obs_points = np.sort(
                        np.random.choice(np.arange(seqlen), size=length, replace=False)
                    )
                    subsampled_mask[i, obs_points, :] = 1
            else:
                subsampled_mask = utils.subsample_timepoints(
                    train_batch[:, :, dim:2 * dim].clone(),
                    args.sample_tp,
                    shuffle=args.shuffle,
                )
            if args.recon_loss or args.sample_tp == 1.0:
                recon_mask = train_batch[:, :, dim:2 * dim]
            else:
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
                num_samples=args.k_iwae,
                beta=kl_coef,
            )
            optimizer.zero_grad()
            loss_info.composite_loss.backward()
            optimizer.step()
            train_loss += loss_info.composite_loss.item() * batch_len
            avg_loglik += loss_info.loglik * batch_len
            avg_kl += loss_info.kl * batch_len
            mse += loss_info.mse * batch_len
            mae += loss_info.mae * batch_len
            train_n += batch_len
        # log scalars to tensorboard
        writer.add_scalar('train_loss_train', train_loss / train_n, itr)
        writer.add_scalar('-avg_loglik_train', -avg_loglik / train_n, itr)
        writer.add_scalar('avg_kl_train', avg_kl / train_n, itr)
        writer.add_scalar('mse_train', mse / train_n, itr)
        writer.add_scalar('mae_train', mae / train_n, itr)
        writer.add_scalar('kl_coeff_train', kl_coef, itr)

        # print train stats to console 100 times
        if itr % int(args.niters / 100) == 0:
            print(
                'Iter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg kl: {:.4f}, '
                'mse: {:.6f}, mae: {:.6f}'.format(
                    itr,
                    train_loss / train_n,
                    -avg_loglik / train_n,
                    avg_kl / train_n,
                    mse / train_n,
                    mae / train_n
                )
            )

        # print val and test stats 40 times
        if itr % int(args.niters / 40) == 0:
            for loader, num_samples, name in [(val_loader, 5, 'val'), (test_loader, 100, 'test')]:
                print('   ', name + ':\t', end='')
                m_avg_loglik, mse, mae, mean_mse, mean_mae = utils.evaluate_hetvae(net, dim, loader, 0.5, shuffle=False, k_iwae=num_samples)
                # writer.add_scalar('train_loss' + '_' + name, train_loss / train_n, itr)
                writer.add_scalar('-avg_loglik' + '_' + name, m_avg_loglik, itr)
                writer.add_scalar('avg_kl' + '_' + name, avg_kl / train_n, itr)
                writer.add_scalar('mse' + '_' + name, mse, itr)
                writer.add_scalar('mae' + '_' + name, mae, itr)
        
        # save model 20 times
        if itr % int(args.niters / 20) == 0 and args.save:
            print('Saved model.')
            save_path = log_dir + str(experiment_id) + '.h5'
            torch.save({
                'args': args,
                'epoch': itr,
                'state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / train_n,
            }, save_path)



class HETVAE():
    def __init__(self, n_features, union_tp, model_args, log_path=None, verbose=True) -> None:
        self.verbose = verbose
        self.parse_flag = False
        self.union_tp = union_tp
         # parse arguments
        self.parse_arguments(model_args=model_args)
        # create log_path
        start_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        log_path = Path(log_path) if log_path is not None else Path.home().joinpath('Documents/sepsis/imputation/runs')
        self.log_path = log_path.joinpath(f'{self.args.dataset}/hetvae/{start_time}/')
        print(f'Logging and saving model to: {self.log_path}')
        # NUmber of features / variables (also called dim by mTAN author)
        self.n_features = n_features
        # set up CUDA or device for torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'self.device: {self.device}')
        # set up model
        self.set_up_model()
        # set up optimizer
        self.set_up_optimizer()
        self.epoch = 0
        # set up tensorboard logging
        self.set_up_tensorboard(self.log_path)
        # load pretrained model
        if self.args.fname is not None:
            self.load_from_checkpoint(self.args.fname)
        # add hparams to dict from args
        self.hparams = vars(self.args)

    def parse_arguments(self, model_args:str):
        parser = argparse.ArgumentParser()
        parser.add_argument('--niters', type=int, default=2000, help='Training epochs or iterations.')
        parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        parser.add_argument('--latent-dim', type=int, default=32, help='Dimension of latent states. The latent representation consists of a mean and a logvar for each latent dimension.')
        parser.add_argument('--rec-hidden', type=int, default=32, help='???')
        parser.add_argument('--width', type=int, default=50, help='???')
        parser.add_argument('--embed-time', type=int, default=128, help='???')
        parser.add_argument('--num-ref-points', type=int, default=128, help='???')
        parser.add_argument('--k-iwae', type=int, default=10, help='??? Number of samples drawn from latent space.')
        parser.add_argument('--save', action='store_true', help='Flag for saving trained model.')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
        parser.add_argument('--n', type=int, default=8000, help='Number of data points to be created. Only for synthetic dataset.')
        parser.add_argument('--batch-size', type=int, default=50, help='Batch size.')
        parser.add_argument('--norm', action='store_true', help='???')
        parser.add_argument('--kl-annealing', action='store_true', help='Annealing (=scheduling) for KL weighting.')
        parser.add_argument('--kl-zero', action='store_true', help='If set, the KL weighting is zero in the loss function.')
        parser.add_argument('--enc-num-heads', type=int, default=1, help='???')
        parser.add_argument('--dataset', type=str, default='toy', help='Name of Dataset to be used.')
        parser.add_argument('--dropout', type=float, default=0.0, help='Dropout regularization value between [0,1) .')
        parser.add_argument('--intensity', action='store_true', help='???')
        parser.add_argument('--net', type=str, default='hetvae', help='Which deep learning model to use.')
        parser.add_argument('--const-var', action='store_true', help='???')
        parser.add_argument('--var-per-dim', action='store_true', help='???')
        parser.add_argument('--std', type=float, default=0.1, help='???')
        parser.add_argument('--sample-tp', type=float, default=0.5, help='How many timepoints to be subsampled. Fraction value between (0,1] .')
        parser.add_argument('--bound-variance', action='store_true', help='???')
        parser.add_argument('--shuffle', action='store_true', help='??? Something with shuffling time points.')
        parser.add_argument('--recon-loss', action='store_true', help='??? Something with sub-sampling time points.')
        parser.add_argument('--normalize-input', type=str, default='znorm', help='???')
        parser.add_argument('--mse-weight', type=float, default=0.0, help='???')
        parser.add_argument('--elbo-weight', type=float, default=1.0, help='???')
        parser.add_argument('--mixing', type=str, default='concat', help='How the deterministic and probabilistic paths of encoder are used. Options are concat_and_mix, concat, seperate, interp_only and na.')
        parser.add_argument('--fname', type=str, default=None, help='Path to loading pretrained model.')
        self.parser = parser
        self.args = self.parser.parse_args(model_args.split())
        self.parse_flag = True
        print(f'Model args: {self.args}')
        return

    def set_up_model(self, args=None):
        self.net = models.load_network(self.args, self.n_features, self.union_tp)
        return

    def set_up_optimizer(self, optimizer=None):
        params = list(self.net.parameters())
        self.optimizer = optim.Adam(params, lr=self.args.lr)
        print('parameters:', utils.count_parameters(self.net))
        return

    def set_up_tensorboard(self, path:Path):
        self.writer = SummaryWriter(log_dir=path)
        # with two plots in one
        layout = {"Training vs. Validation": {
                    "loss": ["Multiline", ["train_loss", "val_loss"]],
                    "mae": ["Multiline", ["mae_train", "mae_val"]],
                    "mse": ["Multiline", ["mse_train", "mse_val"]],
                    "avg_kl": ["Multiline", ["avg_kl_train", "avg_kl_val"]],
                    "-avg_loglik": ["Multiline", ["-avg_loglik_train", "-avg_loglik_val"]], 
                    },
        }
        self.writer.add_custom_scalars(layout)
        print(f'Tensorboard logging to: {path}')
        return

    def load_from_checkpoint(self, path, log_path=None):
        print('\n========================================================')
        print("Loading model from ", path)
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        print('Loading saved weights done. Model trained until epoch ', checkpoint['epoch'])
        if log_path is not None:
            self.set_up_tensorboard(log_path)
        print('========================================================\n')
        return

    def train_model(self, train_loader, val_loader, train_extra_epochs=0):
        args = self.args
        net = self.net
        writer = self.writer
        writer.add_text('hparams', str(self.args), self.epoch)
        self.net.train()
        start = int(self.epoch)
        end = args.niters if train_extra_epochs==0 else self.epoch+train_extra_epochs
        for itr in range(start, end):
            train_loss = 0
            train_n = 0
            avg_loglik, avg_kl, mse, mae = 0, 0, 0, 0
            if args.kl_annealing:
                wait_until_kl_inc = 10000
                if itr < wait_until_kl_inc:
                    kl_coef = 0.
                else:
                    kl_coef = (1 - 0.999 ** (itr - wait_until_kl_inc))
            elif args.kl_zero:
                kl_coef = 0
            else:
                kl_coef = 1
            for train_batch in train_loader:
                batch_len = train_batch.shape[0]
                train_batch = train_batch.to(self.device)
                if args.dataset == 'toy':
                    subsampled_mask = torch.zeros_like(
                        train_batch[:, :, self.n_features:2 * self.n_features]).to(self.device)
                    seqlen = train_batch.size(1)
                    for i in range(batch_len):
                        length = np.random.randint(low=3, high=10)
                        obs_points = np.sort(
                            np.random.choice(np.arange(seqlen), size=length, replace=False)
                        )
                        subsampled_mask[i, obs_points, :] = 1
                else:
                    subsampled_mask = utils.subsample_timepoints(
                        train_batch[:, :, self.n_features:2 * self.n_features].clone(),
                        args.sample_tp,
                        shuffle=args.shuffle,
                    )
                if args.recon_loss or args.sample_tp == 1.0:
                    recon_mask = train_batch[:, :, self.n_features
                    :2 * self.n_features]
                else:
                    recon_mask = train_batch[:, :, self.n_features:2 * self.n_features] - subsampled_mask
                context_y = torch.cat((
                    train_batch[:, :, :self.n_features] * subsampled_mask, subsampled_mask
                ), -1)

                loss_info = net.compute_unsupervised_loss(
                    train_batch[:, :, -1],
                    context_y,
                    train_batch[:, :, -1],
                    torch.cat((
                        train_batch[:, :, :self.n_features] * recon_mask, recon_mask
                    ), -1),
                    num_samples=args.k_iwae,
                    beta=kl_coef,
                )
                self.optimizer.zero_grad()
                loss_info.composite_loss.backward()
                self.optimizer.step()
                train_loss += loss_info.composite_loss.item() * batch_len
                avg_loglik += loss_info.loglik * batch_len
                avg_kl += loss_info.kl * batch_len
                mse += loss_info.mse * batch_len
                mae += loss_info.mae * batch_len
                train_n += batch_len

            self.epoch += 1
            # log scalars to tensorboard
            writer.add_scalar('train_loss_train', train_loss / train_n, itr)
            writer.add_scalar('-avg_loglik_train', -avg_loglik / train_n, itr)
            writer.add_scalar('avg_kl_train', avg_kl / train_n, itr)
            writer.add_scalar('mse_train', mse / train_n, itr)
            writer.add_scalar('mae_train', mae / train_n, itr)
            writer.add_scalar('kl_coeff_train', kl_coef, itr)

            # print train stats to console every epoch
            if (itr == 1) or (itr % 1 == 0):
                print(
                    '{:2.0%} Iter: {}, train loss: {:.4f}, avg nll: {:.4f}, avg kl: {:.4f}, '
                    'mse: {:.6f}, mae: {:.6f}'.format(
                        (itr-start)/(end-start),
                        itr,
                        train_loss / train_n,
                        -avg_loglik / train_n,
                        avg_kl / train_n,
                        mse / train_n,
                        mae / train_n
                    )
                )

            # calculate and print val and test stats every 2 epochs
            if (itr == 1) or (itr % 2 == 0) or (itr == args.niters):
                for loader, num_samples, name in [(val_loader, 5, 'val')]:
                    print('   ', name + ':\t', end='')
                    m_avg_loglik, mse, mae, mean_mse, mean_mae = utils.evaluate_hetvae(net, self.n_features, loader, 0.5, shuffle=False, k_iwae=num_samples)
                    # writer.add_scalar('train_loss' + '_' + name, train_loss / train_n, itr)
                    writer.add_scalar('-avg_loglik' + '_' + name, m_avg_loglik, itr)
                    writer.add_scalar('avg_kl' + '_' + name, avg_kl / train_n, itr)
                    writer.add_scalar('mse' + '_' + name, mse, itr)
                    writer.add_scalar('mae' + '_' + name, mae, itr)
            
            # save model every 5 epochs
            if (itr % 5 == 0) or (itr == args.niters) and args.save:
                print('Saved model.')
                save_path = self.log_path.joinpath('hetvae.h5')
                torch.save({
                    'args': args,
                    'epoch': itr,
                    'state_dict': net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss / train_n,
                }, save_path)
        # Free GPU storage, that might remain reserved otherwise
        torch.cuda.empty_cache()


    def impute(self, batch, num_samples, sample_tp=0.5, shuffle=False):
        dim = self.n_features
        self.net.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            subsampled_mask = utils.subsample_timepoints(
                batch[:, :, dim:2 * dim].clone(),
                sample_tp,
                shuffle=shuffle,
            )
            recon_mask = batch[:, :, dim:2 * dim] - subsampled_mask
            context_y = torch.cat((
                batch[:, :, :dim] * subsampled_mask, subsampled_mask
            ), -1)


            # from compute_unsupervised_loss
            context_x, context_y, target_x, target_y, num_samples = batch[:, :, -1], context_y, batch[:, :, -1], torch.cat((batch[:, :, :dim] * recon_mask, recon_mask), -1), self.args.k_iwae

            # Get output of decoder: px (and encoder: qz)
            px, qz = self.net.get_reconstruction(context_x, context_y, target_x, num_samples)
            pred_mean = px.mean.cpu().numpy()
            # Convert from logarithmic variance to variance or std. deviation
            pred_std = torch.exp(0.5 * px.logvar).cpu().numpy()
            # draw samples and multiply with stdandard deviation and add mean
            preds = np.random.randn(num_samples // 2, num_samples, pred_mean.shape[1], pred_mean.shape[2], pred_mean.shape[3]) * pred_std + pred_mean
            # reshape so preds are of shape: [num_samples, batch_size, time_points, features]
            preds = preds.reshape(-1, pred_mean.shape[1], pred_mean.shape[2], pred_mean.shape[3])
            # calc mean, with shape: [num_samples, batch_size, time_points, features]
            pred_mean = preds.mean(0)
            # calc quantiles
            std_quantile = 0.68  # In normal distribution 1*std. dev from mean includes 68% of the set.
            quantile_high = np.quantile(preds, std_quantile, axis=0)
            quantile_low  = np.quantile(preds, 1-std_quantile, axis=0)
            # free memory that otherwise stays blocked on GPU
            torch.cuda.empty_cache()

            return pred_mean, preds, quantile_low, quantile_high
