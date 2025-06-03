import torch, wandb
import lightning as L
import scanpy as sc
from torch.optim import Adam
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_f1_score
import matplotlib.pyplot as plt

class MesenchymalStates(L.LightningModule):
    def __init__(self, hparams, model):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = model

    def forward(self, x, deterministic = False):
        return self.model(x, deterministic)

    def compute_metrics(self, y, c, w, y_pred, c_logits, mu, logvar):
        clf_rand = torch.log(torch.tensor(c_logits.size(-1), dtype = torch.float32, device = self.device))
        metric_dict = {
            'reg_loss'    : 1 - F.cosine_similarity(y_pred, y, dim = -1),
            'clf_loss'    : F.cross_entropy(c_logits, c, reduction = 'none') / clf_rand,
            'kldiv_loss'  : -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1),
            'mse_loss'    : F.mse_loss(y_pred, y, reduction = 'none').mean(dim = -1),
            'clf_f1score' : multiclass_f1_score(c_logits, c, c_logits.size(-1))
            }
        metric_dict['pred_loss'] = metric_dict['reg_loss'] + metric_dict['clf_loss']
        metric_dict['total_loss'] = w * (metric_dict['pred_loss'] + self.hparams.lambda_kldiv * metric_dict['kldiv_loss'])
        return {key : val.mean() if 'loss' in key else val for key, val in metric_dict.items()}

    def training_step(self, batch, _):
        outs = self.forward(batch[0])
        metric_dict = self.compute_metrics(*batch[1:], *outs[:-1])
        for key in metric_dict:
            self.log(
                f'train_{key}',
                metric_dict[key],
                on_step = False,
                on_epoch = True,
                batch_size = batch[0].size(0),
                sync_dist = True,
                add_dataloader_idx = False
                )
        return metric_dict['total_loss']
    
    def validation_step(self, batch, _):
        outs = self.forward(batch[0], deterministic = True)
        metric_dict = self.compute_metrics(*batch[1:], *outs[:-1])
        for key in metric_dict:
            self.log(
                f'val_{key}',
                metric_dict[key],
                on_step = False,
                on_epoch = True,
                batch_size = batch[0].size(0),
                sync_dist = True,
                add_dataloader_idx = False
                )

    def on_validation_epoch_end(self):
        if (self.current_epoch > 0) and (self.current_epoch % self.hparams.val_plot_freq == 0):
            adata  = self.trainer.val_dataloaders.dataset.adata
            X = next(iter(self.trainer.val_dataloaders))[0].to(self.device) 
            adata.obsm['X_latent'] = self.forward(X, deterministic = True)[-1].detach().cpu().numpy()

            # plot celltype
            fig, ax = plt.subplots(1, 1, figsize = (7, 7))
            sc.pl.embedding(adata, 'X_latent', color = 'celltype', size = 30, legend_loc = 'on data', ax = ax, show = False)
            ax.set_box_aspect(1); fig.tight_layout()
            self.logger.experiment.log({'val_latent_celltype' : wandb.Image(fig), 'epoch' : self.current_epoch})
            plt.close(fig)

            # plot signatures
            colors = [col for col in adata.obs.columns if 'signature' in col]
            sc.pl.embedding(adata, 'X_latent', color = colors, cmap = 'seismic', vcenter = 0, show = False)
            self.logger.experiment.log({'val_latent_signatures' : wandb.Image(plt.gcf()), 'epoch' : self.current_epoch})
            plt.close(plt.gcf())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.hparams.learning_rate)
