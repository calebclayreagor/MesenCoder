import wandb
import lightning as L
import scanpy as sc
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from itertools import combinations

class MesenchymalStates(L.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, y, y_pred, w):
        loss = F.mse_loss(y_pred, y, reduction = 'none').mean(dim = -1)
        return (w * loss).mean()

    def training_step(self, batch, _):
        X, y, w = batch
        y_pred, _ = self.forward(X)
        loss = self.compute_loss(y, y_pred, w)
        self.log(
            'train_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = X.size(0),
            sync_dist = True,
            add_dataloader_idx = False
            )
        return loss
    
    def validation_step(self, batch, _):
        X, y, w = batch
        y_pred, _ = self.forward(X)
        loss = self.compute_loss(y, y_pred, w)
        self.log(
            'val_loss', loss,
            on_step = False,
            on_epoch = True,
            batch_size = X.size(0),
            sync_dist = True,
            add_dataloader_idx = False
            )

    def on_validation_epoch_end(self):
        if (self.current_epoch > 0) and (self.current_epoch % self.hparams.val_plot_freq == 0):
            adata  = self.trainer.val_dataloaders.dataset.adata
            X = next(iter(self.trainer.val_dataloaders))[0].to(self.device)
            adata.obsm['X_latent'] = self.forward(X)[1].detach().cpu().numpy()

            groups = ['Splanchnic Mesoderm',
                      'Posterior Epiblast',
                      'Lateral Plate Mesoderm',
                      'Definitive Endoderm',
                      'Cranial Mesenchyme',
                      'Cranial Neural Crest',
                      'Trunk Neural Crest',
                      'Neuromesodermal Progenitor',
                      'Presomitic Mesoderm',
                      'Premigratory Neural Crest',
                      'Migratory Neural Crest'
                      ]
            colors = [col for col in adata.obs.columns if 'signature' in col]
            
            for dim in combinations(range(self.hparams.latent_dim), 2):
                d0, d1 = dim[0] + 1, dim[1] + 1

                # plot celltype
                fig, ax = plt.subplots(1, 1, figsize = (10, 7))
                sc.pl.embedding(
                    adata, 'X_latent',
                    dimensions = dim,
                    color = 'celltype',
                    groups = groups,
                    size = 100,
                    ax = ax,
                    show = False)
                fig.tight_layout()
                self.logger.experiment.log({
                    f'val_latent_celltype_{d0}{d1}' : wandb.Image(fig),
                    'epoch'                         : self.current_epoch
                    })
                plt.close(fig)

                # plot signatures
                sc.pl.embedding(
                    adata, 'X_latent',
                    dimensions = dim,
                    color = colors,
                    cmap = 'inferno',
                    size = 80,
                    show = False)
                fig = plt.gcf()
                self.logger.experiment.log({
                    f'val_latent_signatures_{d0}{d1}' : wandb.Image(fig),
                    'epoch'                           : self.current_epoch
                    })
                plt.close(fig)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.hparams.learning_rate)

