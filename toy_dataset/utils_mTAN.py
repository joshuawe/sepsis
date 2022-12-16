import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import defaultdict

# from random import SystemRandom
# from imputation.mTAN.src import models, utils
from imputation.models.mTAN import models, utils


class MTAN_ToyDataset():
    def __init__(self, n_features, log_path, model_args=None, verbose=True) -> None:
        self.verbose = verbose
        self.parse_flag = True
        # parse arguments
        self.parse_arguments(model_args=model_args)
        # create log_path
        start_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        self.log_path = log_path + f'{self.args.dataset}/' + f'{start_time}/'
        # NUmber of features / variables (also called dim by mTAN author)
        self.n_features = n_features
        # set up CUDA or device for torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        # set up the log dict
        self.log_dict = self._prepare_log_dict()

        return

    def parse_arguments(self, model_args=None):
        if model_args is None:
            model_args = ['--niters', '100', '--lr', '0.001', '--batch-size', '2', '--rec-hidden', '32', '--latent-dim', '4', '--length', '20', '--enc', 'mtan_rnn', '--dec', 'mtan_rnn', '--n', '1000',  '--gen-hidden', '50', '--save', '1', '--k-iwae', '5', '--std', '0.01', '--norm', '--learn-emb', '--kl', '--seed', '0', '--num-ref-points', '20', '--dataset', 'toy_josh']
            model_args = '--niters 100 --lr 0.001 --batch-size 128 --rec-hidden 32 --latent-dim 4 --length 20 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --save 1 --k-iwae 5 --std 0.01 --norm --learn-emb --kl --seed 0 --num-ref-points 20 --dataset toy'.split()

        parser = argparse.ArgumentParser()
        parser.add_argument('--niters', type=int, default=2000, help='Number of epochs.')
        parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
        parser.add_argument('--std', type=float, default=0.01, help='Unused??? The standard deviation for sampling from latent space.')
        parser.add_argument('--latent-dim', type=int, default=32, help='Size of latent dimension. For each dimension there will be one mean and standard deviation.')
        parser.add_argument('--rec-hidden', type=int, default=32, help='Size of hidden layer in encoder.')
        parser.add_argument('--gen-hidden', type=int, default=50, help='Size of hidden layer in decoder.')
        parser.add_argument('--embed-time', type=int, default=128, help='????')
        parser.add_argument('--k-iwae', type=int, default=10, help='Number of samples to be drawn from latent distribution.')
        parser.add_argument('--save', type=int, default=1, help='Flag, whether the model should be saved.')
        parser.add_argument('--enc', type=str, default='mtan_rnn', help='Which Encoder model should be used.')
        parser.add_argument('--dec', type=str, default='mtan_rnn', help='Which decoder model should be used.')
        parser.add_argument('--fname', type=str, default=None, help='Load pretrained model weights from this path.')
        parser.add_argument('--seed', type=int, default=0, help='Seed for the randomness generator.')
        parser.add_argument('--n', type=int, default=8000, help='Number of generated data points')
        parser.add_argument('--batch-size', type=int, default=50, help='The Batch size.')
        parser.add_argument('--quantization', type=float, default=0.016, help="Quantization on the physionet dataset.")
        parser.add_argument('--classif', action='store_true', help="Include binary classification loss")
        parser.add_argument('--norm', action='store_true', help='???')
        parser.add_argument('--kl', action='store_true', help='Flag, whether KL (or beta) annealing schedule should be used. If not, the KL weight will be a constant 1.')
        parser.add_argument('--learn-emb', action='store_true', help='Learn the time embedding.')
        parser.add_argument('--enc-num-heads', type=int, default=1, help='????')
        parser.add_argument('--dec-num-heads', type=int, default=1, help='????')
        parser.add_argument('--length', type=int, default=20, help='????')
        parser.add_argument('--num-ref-points', type=int, default=128, help='????')
        parser.add_argument('--dataset', type=str, default='toy', help='Name of dataset to be used.')
        parser.add_argument('--enc-rnn', action='store_false', help='????')
        parser.add_argument('--dec-rnn', action='store_false', help='????')
        parser.add_argument('--sample-tp', type=float, default=1.0, help='Percentage of time points that should be sampled.')
        parser.add_argument('--only-periodic', type=str, default=None, help='????')
        parser.add_argument('--dropout', type=float, default=0.0, help='Dropout value for dropout regularization, value in [0,1).')
        self.parser = parser
        self.args = self.parser.parse_args(model_args)
        self.parse_flag = True
        return

    def set_up_model(self, args=None):
        if args is not None:
            self.parse_arguments(args)
        args = self.args
        self.encoder = models.enc_mtan_rnn(
                            self.n_features, 
                            torch.linspace(0, 1., args.num_ref_points), 
                            args.latent_dim, 
                            args.rec_hidden,
                            embed_time=128, 
                            learn_emb=args.learn_emb,
                            num_heads=args.enc_num_heads
        ).to(self.device)

        self.decoder = models.dec_mtan_rnn(
                            self.n_features, 
                            torch.linspace(0, 1., args.num_ref_points), 
                            args.latent_dim, 
                            args.gen_hidden, 
                            embed_time=128, 
                            learn_emb=args.learn_emb, 
                            num_heads=args.dec_num_heads
        ).to(self.device)
        return

    def set_up_optimizer(self, optimizer=None):
        params = (list(self.decoder.parameters()) + list(self.encoder.parameters()))
        self.optimizer = optim.Adam(params, lr=self.args.lr)
        if self.verbose:
            print('parameters encoder/decoder:', utils.count_parameters(self.encoder), utils.count_parameters(self.decoder))
        return

    def set_up_tensorboard(self, path:str):
        # Set up Tensorboard
        self.writer = SummaryWriter(log_dir=path)
        return

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['rec_state_dict'])
        self.decoder.load_state_dict(checkpoint['dec_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        print('Loading saved weights done. Model trained until epoch ', checkpoint['epoch'])
        # print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        # print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 3))
        # print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 10))
        # print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 20))
        # print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 30))
        # print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 50))
        return

    def train_model(self, train_loader, test_loader, train_extra_epochs=None):
        dim = self.n_features
        if train_extra_epochs is not None:
            final_epoch = self.epoch + train_extra_epochs
        else:
            final_epoch = self.args.niters
        print(f'Begin training! Training until epoch {final_epoch}.')
        # Run through epochs
        for itr in range(self.epoch+1, final_epoch+1):
            train_loss = 0
            train_n = 0
            avg_reconst, avg_kl, mse = 0, 0, 0
            if self.args.kl:
                wait_until_kl_inc = int(final_epoch * 0.1)
                if itr < wait_until_kl_inc:
                    kl_coef = 0.
                else:
                    kl_coef = (1 - 0.9 ** (itr - wait_until_kl_inc))
                self.log_scalar('kl_coefficient', kl_coef, itr)
            else:
                kl_coef = 1

            for train_batch in train_loader:
                # get data
                if self.args.dataset != 'XXtoy_josh':
                    train_batch = train_batch.to(self.device)
                    batch_len = train_batch.shape[0]
                    observed_data = train_batch[:, :, :dim]
                    observed_mask = train_batch[:, :, dim:2 * dim]
                    observed_tp = train_batch[:, :, -1]
                else:
                    # send to proper device
                    train_batch = list(train_batch)
                    for i in range(len(train_batch)):
                        train_batch[i] = train_batch[i].to(device)
                    # give proper variable names
                    observed_data, observed_mask, observed_tp, Y = train_batch
                    batch_len = observed_data.shape[0]
                    # concatenate because original implementation requires it
                    train_batch = torch.concatenate(( observed_data, observed_mask, observed_tp.unsqueeze(2)), dim=2)

                if self.args.sample_tp and self.args.sample_tp < 1:
                    subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
                        observed_data.clone(), observed_tp.clone(), observed_mask.clone(), self.args.sample_tp)
                else:
                    subsampled_data, subsampled_tp, subsampled_mask = \
                        observed_data, observed_tp, observed_mask
                # --- Forward pass through encoder --- 
                out = self.encoder(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
                # Interpret the output as mean and log-variance of latent distribution
                qz0_mean = out[:, :, :self.args.latent_dim]
                qz0_logvar = out[:, :, self.args.latent_dim:]
                # --- Sampling from latent distribution --- 
                # draw args.kiwae times from the latent distribution
                # epsilon = torch.randn(qz0_mean.size()).to(device)
                epsilon = torch.randn(
                    self.args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
                ).to(self.device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                # --- forward pass through decoder --- 
                pred_x = self.decoder(
                    z0,
                    observed_tp[None, :, :].repeat(self.args.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
                )
                # nsample, batch, seqlen, dim
                pred_x = pred_x.view(self.args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])
                # --- compute loss --- 
                logpx, analytic_kl = utils.compute_losses(
                    dim, train_batch, qz0_mean, qz0_logvar, pred_x, self.args, self.device)
                loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(self.args.k_iwae))
                # optimizer update step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # compute more losses
                train_loss += loss.item() * batch_len
                train_n += batch_len
                avg_reconst += torch.mean(logpx) * batch_len
                avg_kl += torch.mean(analytic_kl) * batch_len
                mse += utils.mean_squared_error(
                    observed_data, pred_x.mean(0), observed_mask) * batch_len

            print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}'
                .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n))
            self.log_scalar('avg elbo', train_loss / train_n, itr)
            self.log_scalar('avg reconst', -avg_reconst / train_n, itr)
            self.log_scalar('avg kl', avg_kl / train_n, itr)
            self.log_scalar('avg mse', mse / train_n, itr)
            self.epoch += 1

            if itr % 10 == 0:
                mse = utils.evaluate(dim, self.encoder, self.decoder, test_loader, self.args, 1)
                self.log_scalar('Test MSE', mse, itr)
                print('Test Mean Squared Error', mse)
            if itr % 10 == 0 and self.args.save:
                path_save = self.log_path + self.args.dataset + '_' + self.args.enc + '_' + self.args.dec + '.h5'
                torch.save({
                    'args': self.args,
                    'epoch': itr,
                    'rec_state_dict': self.encoder.state_dict(),
                    'dec_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': -1 * loss,
                }, path_save)
                print("Saved model state.")

    def log_scalar(self, tag, scalar_value, global_step):
        """Writes information for tensorboard as well as into a log dict, so the data can be easily used for analysis and plotting.

        Args:
            tag (str): The description of the information.
            scalar_value (floar): The actual value to be stored.
            global_step (int): The epoch of training. 
        """
        self.writer.add_scalar(tag, scalar_value, global_step)
        self.log_dict[tag].append((global_step, scalar_value))

    def _prepare_log_dict(self):
        """Prepares as log dict with all required keys, so that training information can be added and stored. New information should be added by using the `self.log_scalar()` function. The entries are then `{key1:[(val1,epoch1), (val2, epoch2), ...], key2:[...], ...}`. As the log dict is of the class `defaultdict(list)`, values can be appended to the list of a key, even if the key does not exist yet.

        Returns:
            dict: The log dict with empty entries.
        """
        # log_dict = dict()
        # keys = ['avg elbo', 'avg reconst', 'avg kl', 'avg mse', 'kl_coefficient', 'Test MSE']
        # for key in keys:
        #     log_dict[key] = list()
        log_dict = defaultdict(list)
        return log_dict

    def combine_sample(self, X, ind_mask, time_pts):
        """Take the individual matrices and combine / concatenate them to one tensor. All NAN values are converted to zero, because mTAN needs it this way.

        Args:
            X (array): Data with missingness.
            ind_mask (array): Indicating mask for missingness.
            time_pts (array): All time points.
        """
        observed_data = torch.tensor(X)
        observed_mask = torch.tensor(ind_mask)
        observed_tp = torch.tensor(time_pts)
        batch_sample = torch.concatenate((observed_data, observed_mask, observed_tp.unsqueeze(1)), dim=1)
        batch_sample = batch_sample.nan_to_num() # Replace all NAN values with zero.
        return batch_sample.type(torch.float32)

    def split_sample(self, batch:torch.Tensor):
        """Takes a batch that is a combined matrix and splits it up into its subcomponents `observed_data`, `observed_mask`, `observed_tp`.

        Args:
            batch (torch.Tensor): The combined tensor, which should be a batch.

        Returns:
            _type_: observed_data, observed_mask, observed_tp
        """
        if len(batch.shape) != 3:
            raise RuntimeError(f'The given batch is not a batch. Expected a 3 dimensional tensor, instead got shape: {batch.shape}')
        dim = self.n_features
        observed_data = batch[:, :, :dim]
        observed_mask = batch[:, :, dim: 2 * dim]
        observed_tp = batch[:, :, -1]
        return observed_data, observed_mask, observed_tp

    def impute(self, test_batch, k_samples, mean=True):
        """Impute the missingness away from a `test_batch`. The number of draws are `k_samples`.

        Args:
            test_batch (_type_): The testbatch to be imputed.
            k_samples (int): Number of samples to be drawn from the latent distribution.

        Returns:
            _type_: The data with imputed values.
        """
        args = self.args
        dim = self.n_features
        with torch.no_grad():
            test_batch = test_batch.to(self.device)
            observed_data, observed_mask, observed_tp = self.split_sample(test_batch)
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            # forward pass to encoder
            out = self.encoder(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean, qz0_logvar = (
                out[:, :, : args.latent_dim],
                out[:, :, args.latent_dim:],
            )
            # drar k_sample from normal distr. with 
            epsilon = torch.randn(
                k_samples, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(self.device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            batch, seqlen = observed_tp.size()
            time_steps = (
                observed_tp[None, :, :].repeat(k_samples, 1, 1).view(-1, seqlen)
            )
            # forward pass to decoder
            pred_x = self.decoder(z0, time_steps)
            pred_x = pred_x.view(k_samples, -1, pred_x.shape[1], pred_x.shape[2])
            # get the average/mean of all num_sample draws
            if mean is True:
                pred_x = pred_x.mean(0)

        return pred_x




    



    
