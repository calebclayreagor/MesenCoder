import torch
import wandb
import lightning as L
import scanpy as sc
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MesenchymalStates(L.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model

    def forward(self, x, deterministic = False):
        return self.model(x, deterministic)

    def compute_loss(self, y, c, w, y_pred, c_pred, mu, logvar):
        loss_dict = {
            'mse_loss'   : F.mse_loss(y_pred, y, reduction = 'none').mean(-1),
            'clf_loss'   : F.cross_entropy(c_pred, c, reduction = 'none'),
            'kldiv_loss' : -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1),
            }
        pred_loss = ((1 - self.hparams.lambda_clf) * loss_dict['mse_loss']) + \
                    (self.hparams.lambda_clf * loss_dict['clf_loss'])
        loss_dict['loss'] = (w * pred_loss) + (self.hparams.lambda_kldiv * loss_dict['kldiv_loss'])
        return {key : val.mean() for key, val in loss_dict.items()}

    def training_step(self, batch, _):
        outs = self.forward(batch[0])
        loss_dict = self.compute_loss(*batch[1:], *outs[:-1])
        for key in loss_dict:
            self.log(
                f'train_{key}',
                loss_dict[key],
                on_step = False,
                on_epoch = True,
                batch_size = batch[0].size(0),
                sync_dist = True,
                add_dataloader_idx = False
                )
        return loss_dict['loss']
    
    def validation_step(self, batch, _):
        outs = self.forward(batch[0], deterministic = True)
        loss_dict = self.compute_loss(*batch[1:], *outs[:-1])
        for key in loss_dict:
            self.log(
                f'val_{key}',
                loss_dict[key],
                on_step = False,
                on_epoch = True,
                batch_size = batch[0].size(0),
                sync_dist = True,
                add_dataloader_idx = False
                )

    def on_validation_epoch_end(self):
        X = next(iter(self.trainer.val_dataloaders))[0]
        X = X.to(self.device)
        z = self.forward(X, deterministic = True)[-1]
        adata  = self.trainer.val_dataloaders.dataset.adata    
        adata.obsm['X_latent'] = z.detach().cpu().numpy()

        for color in ('celltype', 'Source'):
            fig, ax = plt.subplots(1, 1, figsize = (10, 9.33))
            sc.pl.embedding(adata, 'X_latent', color = color, size = 10, ax = ax, show = False)
            ax.set_box_aspect(1)
            fig.tight_layout()
            self.logger.experiment.log({f'val_embedding_{color}' : wandb.Image(fig), 'epoch' : self.current_epoch})
            plt.close(fig)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.hparams.lr)