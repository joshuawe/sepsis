import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from random import SystemRandom
from imputation.mTAN.src import models, utils


class MTAN_ToyDataset():
    def __init__(self, model_args, n_features, log_path, verbose=True) -> None:
        self.verbose = verbose
        self.log_path = log_path
        # parse arguments
        self.parse_arguments(model_args=model_args)
        # NUmber of features / variables (also called dim by mTAN author)
        self.n_features =n_features
        # set up CUDA or device for torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # set up model
        self.set_up_model()
        # set up optimizer
        self.set_up_optimizer()
        self.epoch = 0
        # set up tensorboard logging
        self.set_up_tensorboard()
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

    def set_up_tensorboard(self):
        # Set up Tensorboard
        start_time = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        path = self.log_path
        path += f'{self.args.dataset}/'
        path += f'{start_time}/'
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


    



    
