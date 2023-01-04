from toy_dataset import data_utils
import matplotlib.pyplot as plt
import numpy as np
import torch 


if __name__ == '__main__':
    print('Start!')
    name = 'toydataset_50000'
    path = data_utils.datasets_dict[name]
    dataset = data_utils.ToyDataDf(path)
    dataset.create_mcar_missingness(0.5)


    print('Loaded data and created missingness.')
    print(f'Cuda memory: {torch.cuda.memory_allocated() / 1024**3:.3f} GB')
    print('--')

 


    name = 'mTAN'
    # model_args='--niters 100 --lr 0.001 --batch-size 128 --rec-hidden 32 --latent-dim 8 --length 40 --enc mtan_rnn --dec mtan_rnn --gen-hidden 50 --save 0 --k-iwae 5 --std 0.01 --norm --learn-emb --kl --seed 3 --num-ref-points 40 --dataset toy'.split()
    model_args = '--niters 100 --lr 0.0001 --batch-size 128 --rec-hidden 256 --latent-dim 32 --length 40 --enc mtan_rnn --dec mtan_rnn --gen-hidden 256 --save 1 --k-iwae 10 --std 0.11 --norm --learn-emb --kl --seed 0 --num-ref-points 40 --dataset toy'.split()
    train_dataloader, validation_dataloader = dataset.prepare_mtan(model_args=model_args, batch_size=100)
    # dataset.mtan.parse_arguments(model_args=model_args)
    path_checkpoint = '/home2/joshua.wendland/Documents/sepsis/runs/mTANtoy/2022.12.19-11.01.28/toy_mtan_rnn_mtan_rnn.h5'
    log_path = '/home2/joshua.wendland/Documents/sepsis/runs/mTANtoy/2022.12.19-11.01.28/'
    # dataset.mtan.load_from_checkpoint(path_checkpoint, log_path=log_path)  

    dataset.train_mtan(train_dataloader, validation_dataloader, epochs=100)

    print('Done!')