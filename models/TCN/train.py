""" Training setup for the TCN model. """

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import wandb
import torch
from typing import Optional, Any, Dict

from .TCN_lightning import TCN_lightning
from .metrics import FinalMetricsCallback


def TCN_fit_pipeline(
        train_dataloader, 
        val_dataloader, 
        input_features: int, 
        project_name: str, 
        pos_weight: Optional[torch.Tensor] = None, 
        config: Optional[Dict[str, Any]]=None
        ) -> None:
    
    if config is None:
        config = {
            'seed': 65,
            'model_type': 'TCN', 
            'time excluded': (input_features==4),
            'num_inputs': input_features,
            'num_channels': [16,16],
            'kernel_size': 2,
            'dropout': 0.0,
            'lr': 0.01,
            'max_epochs': 1000
            }
        
    # add stats
    config['train_samples'] = len(train_dataloader.dataset)
    if val_dataloader is not None:
        config['val_samples'] = len(val_dataloader.dataset)
        
    pl.seed_everything(config['seed'])
    torch.set_float32_matmul_precision('medium')


    # Logger
    wandb_logger = WandbLogger(project=project_name, log_model=True, config=config) #, save_dir="~/Documents/data/ts/synthetic_2/fully_observed/wandb/")
    logger = [wandb_logger]

    # Model
    pl_model = TCN_lightning(config, pos_weight=pos_weight)
    print(pl_model)

    # Callbacks
    es_callback = EarlyStopping(monitor='train_loss', patience=60, mode='min', check_finite=True, min_delta=0.0001, verbose=False)
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    metrics_callback = FinalMetricsCallback()
    callbacks = [es_callback, lr_callback, metrics_callback]

    # trainer
    trainer = pl.Trainer(max_epochs=config['max_epochs'], accelerator='auto', callbacks=callbacks, logger=logger)
    trainer.fit(model=pl_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    print('Finished fit')
    wandb.finish()
    
    return