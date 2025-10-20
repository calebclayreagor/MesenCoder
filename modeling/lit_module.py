import argparse
import numpy as np
import torch, wandb
import lightning as L
import seaborn as sns
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from model import MesenCoder

class MesenchymalStates(L.LightningModule):
    def __init__(self, 
                 hparams: argparse.Namespace,
                 out_pth: str | None = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = MesenCoder(
            n_feature = self.hparams.n_feature,
            n_source = self.hparams.n_source,
            hidden_dim = self.hparams.hidden_dim,
            latent_dim_src = self.hparams.latent_dim_src)
        self.out_pth = out_pth

    def forward(self,
                x: torch.Tensor,
                src: torch.Tensor
                ) -> tuple[torch.Tensor,
                           torch.Tensor]:
        return self.model(x, src)

    def custom_step(self, 
                    batch: tuple[torch.Tensor,
                                 torch.Tensor,
                                 torch.Tensor]
                    ) -> tuple[torch.Tensor,
                               torch.Tensor]:
        X, src, _ = batch
        X_hat, z = self.forward(X, src)
        loss = F.mse_loss(X_hat, X)
        return loss, z

    def training_step(self,
                      batch: tuple[torch.Tensor,
                                   torch.Tensor,
                                   torch.Tensor],
                      _) -> torch.Tensor:
        loss, _ = self.custom_step(batch)
        batch_size = batch[0].size(0)
        self.log(
            'train_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = batch_size,
            sync_dist = True,
            add_dataloader_idx = False)
        return loss
    
    def validation_step(self,
                        batch: tuple[torch.Tensor,
                                     torch.Tensor,
                                     torch.Tensor],
                        _) -> None:
        _, _, ix = batch
        loss, z = self.custom_step(batch)
        batch_size = batch[0].size(0)
        self.log(
            'val_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = batch_size,
            sync_dist = True,
            add_dataloader_idx = False)
        ix = ix.detach().cpu().numpy().squeeze()
        z = z.detach().cpu().numpy().squeeze()
        self.val_latent_z[ix] = z
        
    def predict_step(self,
                     batch: tuple[torch.Tensor,
                                  torch.Tensor,
                                  torch.Tensor],
                        _) -> None:
        _, _, ix = batch
        _, z = self.custom_step(batch)
        ix = ix.detach().cpu().numpy().squeeze()
        z = z.detach().cpu().numpy().squeeze()
        self.pred_latent_z[ix] = z

    def on_validation_epoch_start(self) -> None:
        self.val_latent_z = np.zeros(
            len(self.trainer.val_dataloaders.dataset),
            dtype = np.float32)
        
    def on_predict_epoch_start(self) -> None:
        self.pred_latent_z = np.zeros(
            len(self.trainer.predict_dataloaders.dataset),
            dtype = np.float32)

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch > 0 and (self.current_epoch + 1) % self.hparams.val_log_freq == 0:
            adata = self.trainer.val_dataloaders.dataset.adata
            adata.obs['latent_z'] = self.val_latent_z.copy()
            msk_cancer = adata.obs.celltype.isin(['Malignant'])
            
            for name, yvar, hue, msk in (
                ('dev', 'celltype', 'source', ~msk_cancer),
                ('cancer', 'Disease', 'Category', msk_cancer)):

                data = adata[msk].obs.copy()
                order = (data.groupby(yvar)
                         .agg({'latent_z' : 'mean', hue : 'first'})
                         .sort_values([hue, 'latent_z']).index)
                
                fig, ax = plt.subplots(1, 1, figsize = (8, 10))
                sns.violinplot(
                    data = data,
                    x = 'latent_z',
                    y = yvar,
                    hue = hue,
                    dodge = False,
                    order = order,
                    density_norm = 'width',
                    inner = 'quart',
                    ax = ax)
                ax.grid('x')
                fig.tight_layout()
                
                self.logger.experiment.log({
                    f'val_latent_{name}' : wandb.Image(fig),
                    'epoch'              : self.current_epoch})
                plt.close(fig)

    def on_predict_epoch_end(self) -> None:
        adata = self.trainer.predict_dataloaders.dataset.adata
        adata.obs['latent_z'] = self.pred_latent_z.copy()
        adata.write(self.out_pth)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr = self.hparams.learning_rate)
