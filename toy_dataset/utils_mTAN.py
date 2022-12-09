import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# from random import SystemRandom
from imputation.mTAN.src import models, utils


class MTAN_ToyDataset():
    def __init__(self, n_features, log_path, model_args=None, verbose=True) -> None:
        self.verbose = verbose
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

        return

    def parse_arguments(self, model_args=None):
        if model_args is None:
            model_args = ['--niters', '5000', '--lr', '0.0001', '--batch-size', '2', '--rec-hidden', '32', '--latent-dim', '4', '--length', '20', '--enc', 'mtan_rnn', '--dec', 'mtan_rnn', '--n', '1000',  '--gen-hidden', '50', '--save', '1', '--k-iwae', '5', '--std', '0.01', '--norm', '--learn-emb', '--kl', '--seed', '0', '--num-ref-points', '20', '--dataset', 'toy_josh']

        parser = argparse.ArgumentParser()
        parser.add_argument('--niters', type=int, default=2000, help='Number of epochs')
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--std', type=float, default=0.01)
        parser.add_argument('--latent-dim', type=int, default=32)
        parser.add_argument('--rec-hidden', type=int, default=32)
        parser.add_argument('--gen-hidden', type=int, default=50)
        parser.add_argument('--embed-time', type=int, default=128)
        parser.add_argument('--k-iwae', type=int, default=10)
        parser.add_argument('--save', type=int, default=1)
        parser.add_argument('--enc', type=str, default='mtan_rnn')
        parser.add_argument('--dec', type=str, default='mtan_rnn')
        parser.add_argument('--fname', type=str, default=None)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--n', type=int, default=8000, help='Number of generated data points')
        parser.add_argument('--batch-size', type=int, default=50)
        parser.add_argument('--quantization', type=float, default=0.016,
                            help="Quantization on the physionet dataset.")
        parser.add_argument('--classif', action='store_true',
                            help="Include binary classification loss")
        parser.add_argument('--norm', action='store_true')
        parser.add_argument('--kl', action='store_true')
        parser.add_argument('--learn-emb', action='store_true')
        parser.add_argument('--enc-num-heads', type=int, default=1)
        parser.add_argument('--dec-num-heads', type=int, default=1)
        parser.add_argument('--length', type=int, default=20)
        parser.add_argument('--num-ref-points', type=int, default=128)
        parser.add_argument('--dataset', type=str, default='toy')
        parser.add_argument('--enc-rnn', action='store_false')
        parser.add_argument('--dec-rnn', action='store_false')
        parser.add_argument('--sample-tp', type=float, default=1.0)
        parser.add_argument('--only-periodic', type=str, default=None)
        parser.add_argument('--dropout', type=float, default=0.0)
        self.parser = parser
        self.args = self.parser.parse_args(model_args)
        return

    def set_up_model(self, args=None):
        if args is not None:
            self.pass_arguments(args)
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

    def set_up_tensorboard(self, path):
        # Set up Tensorboard
        self.writer = SummaryWriter(log_dir=path)
        return

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['rec_state_dict'])
        self.decoder.load_state_dict(checkpoint['dec_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 3))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 10))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 20))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 30))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 50))
        return

    def train_model(self, train_loader, test_loader, train_extra_epochs=None):
        dim = self.n_features
        if train_extra_epochs is not None:
            final_epoch = self.epoch + train_extra_epochs
        else:
            final_epoch = self.args.niters
        # Run through epochs
        for itr in range(self.epoch+1, final_epoch+1):
            train_loss = 0
            train_n = 0
            avg_reconst, avg_kl, mse = 0, 0, 0
            if self.args.kl:
                wait_until_kl_inc = int(self.args.niters * 0.4)
                if itr < wait_until_kl_inc:
                    kl_coef = 0.
                else:
                    kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
                self.writer.add_scalar('kl_coefficient', kl_coef, itr)
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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_len
                train_n += batch_len
                avg_reconst += torch.mean(logpx) * batch_len
                avg_kl += torch.mean(analytic_kl) * batch_len
                mse += utils.mean_squared_error(
                    observed_data, pred_x.mean(0), observed_mask) * batch_len

            print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}'
                .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n))
            self.writer.add_scalar('avg elbo', train_loss / train_n, itr)
            self.writer.add_scalar('avg reconst', -avg_reconst / train_n, itr)
            self.writer.add_scalar('avg kl', avg_kl / train_n, itr)
            self.writer.add_scalar('avg mse', mse / train_n, itr)
            self.epoch += 1

            if itr % 10 == 0:
                mse = utils.evaluate(dim, self.encoder, self.decoder, test_loader, self.args, 1)
                self.writer.add_scalar('Test MSE', mse, itr)
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



    



    
