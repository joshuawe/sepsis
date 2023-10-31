# Metrics and plots for classification task


import torch
import pytorch_lightning as pl
import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger



@torch.no_grad()
def get_label_and_preds(model, dataloader: "FastDataLoader", pad_value=-1):
    model.eval()
    # get predictions and targets
    y_true = []
    y_score = []
    # for batch in tqdm(dataloader, desc="Calculating metrics"):
    for batch in dataloader:
        input = batch[:, :, :-1]
        target = batch[:, :, -1]
        pred, _, _ = model(batch)
        target, pred = target.ravel(), pred.ravel()  # shape from (batch, time_pts) to (batch*time_pts)
        idx = (target != pad_value) # only keep time_pts, where there was no padding
        target, pred = target[idx], pred[idx]
        y_true.append(target.detach().numpy())
        y_score.append(pred.detach().squeeze().numpy())
    y_true = np.concatenate(y_true).ravel()
    y_score = np.concatenate(y_score).ravel()
    print('Applying sigmoid to y_pred!')
    y_logits = y_score
    y_score = torch.sigmoid(torch.from_numpy(y_score)).numpy()
    y_pred = (y_score > 0.5) * 1.0
    print(y_true.shape, y_logits.shape, y_score.shape, y_pred.shape)
    return y_true, y_logits, y_score, y_pred




def accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # return np.sum(y_true == y_pred) / len(y_true)
    return np.mean(y_true == y_pred) 



def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> "matplotlib.figure.Figure":
    import matplotlib.colors as colors
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred.round())
    # normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create the ConfusionMatrixDisplay instance and plot it
    cmd = ConfusionMatrixDisplay(cm, display_labels=['class 0\nnegative', 'class 1\npositive'])
    fig, ax = plt.subplots(figsize=(4,4))
    cmd.plot(cmap='YlOrRd', values_format='', colorbar=False, ax=ax, text_kw={'visible':False})
    cmd.texts_ = []
    cmd.text_ = []

    text_labels = ['TN', 'FP', 'FN', 'TP']
    cmap_min, cmap_max = cmd.im_.cmap(0), cmd.im_.cmap(1.0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{text_labels[i * 2 + j]}\n{cmd.im_.get_array()[i, j]:.2%}",
                    ha="center", va="center", color=cmap_min if cmd.im_.get_array()[i, j] > 0.5 else cmap_max)
            
    ax.vlines([0.5], *ax.get_ylim(), color='white', linewidth=1)
    ax.hlines([0.49], *ax.get_xlim(), color='white', linewidth=1)
    ax.spines[:].set_visible(False)
    
    
    bounds = np.linspace(0, 1, 11)
    cmap = plt.cm.get_cmap('YlOrRd', len(bounds)+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    cbar = ax.figure.colorbar(cmd.im_, ax=ax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds[::2], location="right", shrink=0.8)
    # cbar.set_ticks(np.arange(0,1.1,0.1))
    cbar.ax.yaxis.set_ticks_position('both')
    cbar.outline.set_visible(False)
    plt.tight_layout()
    
    return fig




def plot_classification_report(y_test: torch.Tensor, 
                               y_pred: torch.Tensor, 
                               title='Classification Report', 
                               figsize=(8, 6), 
                               save_fig_path=None, **kwargs):
    """
    Plot the classification report of sklearn
    
    Parameters
    ----------
    y_test : pandas.Series of shape (n_samples,)
        Targets.
    y_pred : pandas.Series of shape (n_samples,)
        Predictions.
    title : str, default = 'Classification Report'
        Plot title.
    fig_size : tuple, default = (8, 6)
        Size (inches) of the plot.
    dpi : int, default = 70
        Image DPI.
    save_fig_path : str, defaut=None
        Full path where to save the plot. Will generate the folders if they don't exist already.
    **kwargs : attributes of classification_report class of sklearn
    
    Returns
    -------
        fig : Matplotlib.pyplot.Figure
            Figure from matplotlib
        ax : Matplotlib.pyplot.Axe
            Axe object from matplotlib
    """    
    import matplotlib as mpl
    import matplotlib.colors as colors
    import seaborn as sns
    import pathlib
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cmap = 'YlOrRd'
        
    clf_report = classification_report(y_test, y_pred, output_dict=True, **kwargs)
    keys_to_plot = [key for key in clf_report.keys() if key not in ('accuracy', 'macro avg', 'weighted avg')]
    df = pd.DataFrame(clf_report, columns=keys_to_plot).T
    #the following line ensures that dataframe are sorted from the majority classes to the minority classes
    df.sort_values(by=['support'], inplace=True) 
    
    #first, let's plot the heatmap by masking the 'support' column
    rows, cols = df.shape
    mask = np.zeros(df.shape)
    mask[:,cols-1] = True
    
    bounds = np.linspace(0, 1, 11)
    cmap = plt.cm.get_cmap('YlOrRd', len(bounds)+1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    ax = sns.heatmap(df, mask=mask, annot=False, cmap=cmap, fmt='.3g',
            cbar_kws={'ticks':bounds[::2], 'norm':norm, 'boundaries':bounds},
            vmin=0.0,
            vmax=1.0,
            linewidths=2, linecolor='white'
                    )
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_ticks_position('both')
    
    cmap_min, cmap_max = cbar.cmap(0), cbar.cmap(1.0)
    
    # add text annotation to heatmap
    dx, dy = 0.5, 0.5
    for i in range(rows):
        for j in range(cols-1):
            text = f"{df.iloc[i, j]:.2%}" #if (j<cols) else f"{df.iloc[i, j]:.0f}"
            ax.text(j + dx , i + dy, text,
                    # ha="center", va="center", color='black')
                    ha="center", va="center", color=cmap_min if df.iloc[i, j] > 0.5 else cmap_max)
    
    #then, let's add the support column by normalizing the colors in this column
    mask = np.zeros(df.shape)
    mask[:,:cols-1] = True    
    
    ax = sns.heatmap(df, mask=mask, annot=False, cmap=cmap, cbar=False,
            linewidths=2, linecolor='white', fmt='.0f',
            vmin=df['support'].min(),
            vmax=df['support'].sum(),         
            norm=mpl.colors.Normalize(vmin=df['support'].min(),
                                      vmax=df['support'].sum())
                ) 
    
    cmap_min, cmap_max = cbar.cmap(0), cbar.cmap(1.0)
    for i in range(rows):
        j = cols-1
        text = f"{df.iloc[i, j]:.0f}" #if (j<cols) else f"{df.iloc[i, j]:.0f}"
        color = (df.iloc[i, j]) / (df['support'].sum())
        ax.text(j + dx , i + dy, text,
                # ha="center", va="center", color='black')
                ha="center", va="center", color=cmap_min if color > 0.5 else cmap_max)
            
    plt.title(title)
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 360)
    plt.tight_layout()
         
    if (save_fig_path != None):
        path = pathlib.Path(save_fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, bbox_inches='tight')
    
    return fig, ax





def plot_roc_curve(y_true: torch.Tensor, y_score: torch.Tensor, figsize=(5,5)):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.scatter(fpr, tpr, marker='o', alpha=0.1, facecolors='None', edgecolors='C0')
    plt.fill_between(fpr, tpr, alpha=0.2, color='C0')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlim([0.0, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right", frameon=False)
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle='-', linewidth=0.5, color='grey', alpha=0.5)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    return fig, roc_auc
    
    
    
def y_score_histogram(y_score: torch.Tensor):
    fig, ax = plt.subplots(figsize=(5,5))
    # plt.hist(y_score, density=True, rwidth=0.98)
    sns.histplot(y_score, stat='probability', kde=False, color='C0', ax=ax, bins=25, edgecolor='white')
    ax.spines[:].set_visible(False)
    ax.grid(True, linestyle='-', linewidth=0.5, color='grey', alpha=0.5)
    ax.set(ylabel='Probability [-]', xlabel='y_score [-]', xlim=(-0.001,1.005), ylim=(-0.001,None))
    # ax.set_yticks(np.arange(0, 1.1, 0.2))
    plt.tight_layout()
    return fig




class FinalMetricsCallback(pl.Callback):
    def on_keyboard_interrupt(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.on_fit_end(trainer, pl_module)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        STEP = 1
        if (trainer.test_dataloaders is not None):
            dataloader = trainer.test_dataloaders[0]
            print("Using test_dataloader for final metrics.")
        elif (trainer.val_dataloaders is not None):
            dataloader = trainer.val_dataloaders[0]
            print("Careful, using val_dataloader for final metrics. Please use test_dataloader if you have one.")
        else:
            dataloader = trainer.train_dataloader
            print("Careful, using train_dataloader for final metrics. Please use test_dataloader if you have one.")
            
        # dataloader = trainer.train_dataloader
        # dataloader = trainer.val_dataloaders[0]
        
        # get model outputs
        y_true, y_logits, y_score, y_pred = get_label_and_preds(pl_module, dataloader)
        print(f"y_true: {y_true.shape}, y_logits: {y_logits.shape}, y_score: {y_score.shape}, y_pred: {y_pred.shape}")
        # get accuracy
        accuracy = accuracy_score(y_true, y_pred)
        wandb.run.summary["accuracy"] = accuracy
        # get fpr and tpr
        # fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fig_roc, roc_auc = plot_roc_curve(y_true, y_score)
        trainer.logger.experiment.log({"roc": wandb.Image(fig_roc)})
        # get AUROC
        wandb.run.summary["AUROC"] = np.round(roc_auc, 4)
        # confusion matrix
        fig_cm = plot_confusion_matrix(y_true, y_pred)
        wandb.log({"confusion_matrix": wandb.Image(fig_cm)})
        # classification report
        fig, ax = plot_classification_report(y_true, y_pred, figsize=(7,3))
        trainer.logger.experiment.log({"classification_report": wandb.Image(fig)})
        # histogram of y_scores
        fig = y_score_histogram(y_score)
        trainer.logger.experiment.log({"y_score histogram": wandb.Image(fig)})
        
        # wandb.run.log_artifact()
        print('Accuracy: ', accuracy)
        # wandb.run.log_code()
        # wandb.finish()
        print("AUROC: ", roc_auc)
        plt.close('all')
        return