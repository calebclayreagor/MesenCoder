import argparse
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
            input_dim = self.hparams.input_dim,
            n_source = self.hparams.n_source,
            n_layers_enc = self.hparams.n_layers_enc,
            n_layers_dec = self.hparams.n_layers_dec,
            hidden_dim_enc = self.hparams.hidden_dim_enc,
            hidden_dim_dec = self.hparams.hidden_dim_dec)
        self.out_pth = out_pth

    def forward(self,
                x: torch.Tensor,
                src: torch.Tensor | None = None
                ) -> torch.Tensor:
        return self.model(x, src) 

    def custom_step(self, 
                    batch: tuple[torch.Tensor,
                                 torch.Tensor]
                    ) -> torch.Tensor:
        X, src = batch
        X_hat = self.forward(X, src)
        return F.mse_loss(X_hat, X)

    def training_step(self,
                      batch: tuple[torch.Tensor,
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

                # val latent embedding
                adata = self.trainer.val_dataloaders.dataset.adata
                X, _ = next(iter(self.trainer.val_dataloaders))
                X = X.to(self.device)
                z = self.forward(X)
                adata.obs['latent_z'] = z.detach().cpu().numpy()

                # plot celltype, disease
                msk_cancer = adata.obs.celltype.isin(['Malignant'])
                for yvar, msk, figsize, color in (
                    ('celltype', None, (8, 10), 'cornflowerblue'),
                    ('Disease', msk_cancer, (9, 10.25), 'lightcoral')):
                    if msk is not None:
                        data = adata[msk].obs.copy()
                    else:
                        data = adata.obs.copy()
                    order = (data.groupby(yvar)
                             .latent_z.median()
                             .sort_values().index)
                    fig, ax = plt.subplots(1, 1, figsize = figsize)
                    sns.boxplot(
                        data = data,
                        x = 'latent_z',
                        y = yvar,
                        order = order,
                        color = color,
                        width = .66,
                        fliersize = 0,
                        ax = ax)
                    ax.set_xlim([-1.025, 1.025])
                    fig.tight_layout()
                    self.logger.experiment.log({
                        f'val_latent_{yvar}' : wandb.Image(fig),
                        'epoch'              : self.current_epoch})
                    plt.close(fig)

    # def on_predict_epoch_end(self) -> None:
    #     adata = self.trainer.predict_dataloaders.dataset.adata
    #     X, src, _ = next(iter(self.trainer.predict_dataloaders))
    #     X = X.to(self.device)
    #     src = torch.zeros_like(src, device = self.device)
    #     X_hat, z = self.forward(X, src)
    #     adata.obsm['X_latent'] = z.detach().cpu().numpy()
    #     adata.layers['MesenCoder'] = X_hat.detach().cpu().numpy()
    #     adata.varm['MesenCoder_logvar'] = self.model.logvar_x.detach().cpu().numpy()
    #     adata.varm['MesenCoder_mu'] = self.model.mu_x.detach().cpu().numpy()
    #     adata.varm['MesenCoder_scale'] = self.model.scale_x.detach().cpu().numpy()
    #     adata.write(self.out_pth)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr = self.hparams.learning_rate)
