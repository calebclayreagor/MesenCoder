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

    def training_step(self, batch, _):
        (X1, src1, w1), (X2, src2, w2) = batch
        X1_hat, z1 = self.forward(X1, src1)
        _, z2 = self.forward(X2, src2)
        recon_loss = w1 * F.mse_loss(X1_hat, X1, reduction = 'none').mean(-1)
        w_avg = (w1 + w2) / 2
        pull_loss = w_avg * F.mse_loss(z2, z1, reduction = 'none').mean(-1)
        loss = (recon_loss + self.hparams.lambda_pull * pull_loss).mean()
        self.log(
            'train_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = X1.size(0),
            sync_dist = True,
            add_dataloader_idx = False
            )
        return loss
    
    def validation_step(self, batch, _):
        (X1, src1, w1), (X2, src2, w2) = batch
        X1_hat, z1 = self.forward(X1, src1)
        _, z2 = self.forward(X2, src2)
        recon_loss = w1 * F.mse_loss(X1_hat, X1, reduction = 'none').mean(-1)
        w_avg = (w1 + w2) / 2
        pull_loss = w_avg * F.mse_loss(z2, z1, reduction = 'none').mean(-1)
        loss = (recon_loss + self.hparams.lambda_pull * pull_loss).mean()
        self.log(
            'val_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = X1.size(0),
            sync_dist = True,
            add_dataloader_idx = False
            )

    def on_validation_epoch_end(self):
        if (self.current_epoch > 0) and (self.current_epoch % self.hparams.val_plot_freq == 0):
            adata  = self.trainer.val_dataloaders.dataset.adata
            X, src, _ = next(iter(self.trainer.val_dataloaders))[0]
            X = X.to(self.device)
            src = torch.zeros_like(src, device = self.device)
            _, z = self.forward(X, src)
            adata.obsm['X_latent'] = z.detach().cpu().numpy()
            
            # mesenchymal celltypes
            groups = ['Splanchnic Mesoderm',
                      'Lateral Plate Mesoderm',
                      'Cranial Mesenchyme',
                      'Cranial Neural Crest',
                      'Trunk Neural Crest',
                      'Neuromesodermal Progenitor',
                      'Presomitic Mesoderm',
                      'Premigratory Neural Crest',
                      'Migratory Neural Crest']
            
            # plot celltypes
            fig, ax = plt.subplots(1, 1, figsize = (10, 7))
            group_msk = adata.obs.celltype.isin(groups)
            sc.pl.embedding(
                adata, 'X_latent',
                size = 100,
                ax = ax,
                show = False)
            sc.pl.embedding(
                adata[group_msk],
                'X_latent',
                color = 'celltype',
                groups = groups,
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

