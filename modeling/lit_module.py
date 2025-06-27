import torch
import wandb
import lightning as L
import scanpy as sc
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

class MesenchymalStates(L.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model

    def forward(self, x, src):
        return self.model(x, src)
    
    def custom_step(self, batch):
        X, src, w = batch
        X_hat, _ = self.forward(X, src)
        loss = F.mse_loss(X_hat, X, reduction = 'none')
        loss = w * loss.mean(-1)
        return loss.mean()

    def training_step(self, batch, _):
        loss = self.custom_step(batch)
        batch_size = batch[0].size(0)
        self.log(
            'train_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = batch_size,
            sync_dist = True,
            add_dataloader_idx = False)
        return loss
    
    def validation_step(self, batch, _):
        loss = self.custom_step(batch)
        batch_size = batch[0].size(0)
        self.log(
            'val_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = batch_size,
            sync_dist = True,
            add_dataloader_idx = False)

    def on_validation_epoch_end(self):
        if (self.current_epoch > 0) and (self.current_epoch % self.hparams.val_plot_freq == 0):
            adata  = self.trainer.val_dataloaders.dataset.adata
            X, src, _ = next(iter(self.trainer.val_dataloaders))
            X = X.to(self.device)
            src = torch.zeros_like(src, device = self.device)
            _, z = self.forward(X, src)
            adata.obsm['X_latent'] = z.detach().cpu().numpy()
            
            # plot celltypes
            fig, ax = plt.subplots(1, 1, figsize = (10, 7))
            sc.pl.embedding(
                adata[~adata.obs.trajectory],
                'X_latent',
                size = 100,
                ax = ax,
                show = False)
            sc.pl.embedding(
                adata[adata.obs.trajectory],
                'X_latent',
                color = 'celltype',
                add_outline = True,
                size = 100,
                ax = ax,
                show = False)
            fig.tight_layout()
            self.logger.experiment.log({
                'val_latent_celltype' : wandb.Image(fig),
                'epoch'               : self.current_epoch})
            plt.close(fig)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.hparams.learning_rate)

