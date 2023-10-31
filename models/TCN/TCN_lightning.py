
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# from TCN import TemporalConvNet as TCN
from .TCN import TemporalConvNet as TCN
from .loss import MaskedBCEWithLogitsLoss


class TCN_lightning(pl.LightningModule):
    
    def __init__(self, config: dict, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.config['num_kernel'] = len(config['num_channels'])
        self.loss = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
        self.model = TCN(
                    num_inputs=config['num_inputs'], 
                    num_channels=config['num_channels'], 
                    kernel_size=config['kernel_size'], 
                    dropout=config['dropout']
        )
        self.model.apply(init_weights_relu)
        self.model = self.model.double()
        fc_input = config['num_channels'][-1]
        print('MLP part input size: ', fc_input)
        self.fc = torch.nn.Sequential(*[torch.nn.Linear(fc_input, 1).double()])
        self.save_hyperparameters()
        self.pad_value = -1
        print('Value recognized as pad value:', self.pad_value)
        # Count the number of hyperparameters
        
    def forward(self, batch: torch.Tensor):
        # split into input and target
        input = batch[:, :, :-1]
        target = batch[:, :, -1]
        
        mask = ~(target == self.pad_value)
        mask = mask.to(torch.double)
        input = input.to(torch.double)
        output = self.model(input)
        output = self.fc(output)
        return output, mask, target
    
    def training_step(self, batch, batch_idx):
        output, mask, target = self(batch)
        loss = self.loss(output, target, mask)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output, mask, target = self(batch)
        loss = self.loss(output, target, mask)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['lr'],
            weight_decay = self.config.get('weight_decay', 0)  # 0.0001
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) # Adjust step_size and gamma as needed
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=60, threshold=0.0001, threshold_mode='rel', verbose=True)
        optimizer_dict = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': 'val_loss',
                    'strict': True,
                },
                'monitor': 'train_loss'}
        return optimizer_dict
    
    def plot_losses(self):
        return
    
    def on_fit_start(self):
        """ Execute before training/val/test of Trainer begins."""
        num_hyperparams = torch.sum(torch.tensor([p.numel() for p in self.parameters() if p.requires_grad])).item()
        self.logger.experiment.summary.update({'num_hyperparams': num_hyperparams})
        self.trainer.logger.experiment.watch(self, log="all", log_freq=50)
        return super().on_fit_start()
    
    # def on_fit_end(self) -> None:
    #     y_true, y_logits, y_score, y_pred = get_label_and_preds(pl_model, dataloader)
    #     fig, ax = plot_classification_report(y_true, y_pred, figsize=(7,3))
    #     self.trainer.logger.experiment.summary.update({"gdr456": wandb.Image(fig)})
    #     self.trainer.logger.experiment.summary.update({"grsrg": wandb.Image(fig)})
    #     return super().on_fit_end()
        
def init_weights_relu(m):
    # He Kaiming initialization for ReLU models
    if type(m) == torch.nn.Linear:
        print('Linear layer initialized with Xavier Uniform.')
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == torch.nn.Conv1d:
        print('Conv1d layer initialized with Kaiming Uniform.')
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)
    