import argparse
import torch
import wandb
import lightning as L
import scanpy as sc
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import Optimizer
import matplotlib.pyplot as plt
from model import MesenCoder

class MesenchymalStates(L.LightningModule):
    def __init__(self, 
                 hparams: argparse.Namespace,
                 out_pth: str = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = MesenCoder(
            input_dim = self.hparams.input_dim,
            n_source = self.hparams.n_source,
            hidden_dim = self.hparams.hidden_dim,
            n_layers = self.hparams.n_layers,
            latent_dim = self.hparams.latent_dim)
        self.out_pth = out_pth

    def forward(self, x: torch.Tensor, src: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x, src)
    
    def custom_step(self, 
                    batch: tuple[torch.Tensor,
                                 torch.Tensor,
                                 torch.Tensor]
                    ) -> torch.Tensor:
        X, src, w = batch
        X_hat, _ = self.forward(X, src)
        loss = F.mse_loss(X_hat, X, reduction = 'none')
        loss = w * loss.mean(-1)
        return loss.mean()

    def training_step(self,
                      batch: tuple[torch.Tensor,
                                   torch.Tensor,
                                   torch.Tensor],
                      _) -> torch.Tensor:
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
    
    def validation_step(self,
                        batch: tuple[torch.Tensor,
                                     torch.Tensor,
                                     torch.Tensor],
                        _) -> None:
        loss = self.custom_step(batch)
        batch_size = batch[0].size(0)
        self.log(
            'val_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = batch_size,
            sync_dist = True,
            add_dataloader_idx = False)
        
    def predict_step(self, _, __) -> None:
        return None

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % self.hparams.val_log_freq == 0:
            if self.current_epoch > 0:
                adata = self.trainer.val_dataloaders.dataset.adata
                X, src, _ = next(iter(self.trainer.val_dataloaders))
                X = X.to(self.device)
                src = torch.zeros_like(src, device = self.device)
                _, z = self.forward(X, src)
                adata.obsm['X_latent'] = z.detach().cpu().numpy()
                
                # plot celltypes
                fig, ax = plt.subplots(1, 1, figsize = (10, 7))
                msk_traj = (adata.obs.trajectory == 'True')
                sc.pl.embedding(
                    adata[~msk_traj],
                    'X_latent',
                    size = 100,
                    ax = ax,
                    show = False)
                sc.pl.embedding(
                    adata[msk_traj],
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

    def on_predict_epoch_end(self) -> None:
        adata = self.trainer.predict_dataloaders.dataset.adata
        X, src, _ = next(iter(self.trainer.predict_dataloaders))
        X = X.to(self.device)
        src = torch.zeros_like(src, device = self.device)
        X_hat, z = self.forward(X, src)
        adata.obsm['X_latent'] = z.detach().cpu().numpy()
        adata.layers['MesenCoder'] = X_hat.detach().cpu().numpy()
        adata.varm['MesenCoder_logvar'] = self.model.logvar_x.detach().cpu().numpy()
        adata.varm['MesenCoder_mu'] = self.model.mu_x.detach().cpu().numpy()
        adata.varm['MesenCoder_scale'] = self.model.scale_x.detach().cpu().numpy()
        adata.write(self.out_pth)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr = self.hparams.learning_rate)
