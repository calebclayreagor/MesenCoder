import os, argparse
import lightning as L
import anndata as ad
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from lit_module import MesenchymalStates
from dataset import MesenchymeDataset
from model import MesNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--n_layers', type = int, default = 8)
    parser.add_argument('--hidden_dim', type = int, default = 1024)
    parser.add_argument('--learning_rate', type = float, default = 3e-4)
    parser.add_argument('--batch_size', type = int, default = 32)

    # other
    parser.add_argument('--patience', type = int, default = 10)
    parser.add_argument('--min_delta', type = float, default = 1e-3)
    parser.add_argument('--max_epochs', type = int, default = 100)
    parser.add_argument('--gradient_clip_val', type = float, default = 1.)
    parser.add_argument('--sample_frac', type = float, default = 1.)
    parser.add_argument('--save_ckpt', type = bool, default = False)
    parser.add_argument('--val_plot_freq', type = int, default = 10)
    parser.add_argument('--latent_dim', type = int, default = 3)
    parser.add_argument('--num_workers', type = int, default = 32)
    args = parser.parse_args()

    L.seed_everything(1)

    # training/validation datasets
    adata = ad.read_h5ad(os.path.join('..', 'data', 'modeling', 'training.h5ad'))
    train_ix = (adata.obs.training == 'True')
    adata_train = adata[train_ix]
    adata_val = adata[~train_ix]
    if args.sample_frac < 1:
        train_sample_ix = adata_train.obs.groupby('celltype').sample(frac = args.sample_frac).index
        val_sample_ix = adata_val.obs.groupby('celltype').sample(frac = args.sample_frac).index
        adata_train = adata_train[train_sample_ix]
        adata_val = adata_val[val_sample_ix]
    train_ds = MesenchymeDataset(adata_train)
    val_ds = MesenchymeDataset(adata_val) 

    # dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        pin_memory = True
        )
    val_dl = DataLoader(
        val_ds,
        batch_size = len(val_ds),
        shuffle = False,
        num_workers = 1,
        pin_memory = True
        )

    # encoder -> n-dim latents -> regressor
    model = MesNet(
        input_dim = adata.shape[1],
        target_dim = adata.obsm['X_signature'].shape[1],
        hidden_dim = args.hidden_dim,
        n_layers = args.n_layers,
        latent_dim = args.latent_dim
        )
    lit_model = MesenchymalStates(args, model)

    # wandb logger
    logger = WandbLogger(
        project = 'mesenchymal-states',
        log_model = args.save_ckpt
        )
    logger.watch(lit_model, log = 'all')

    # callbacks
    callbacks = [
        EarlyStopping(
            monitor = 'val_loss',
            patience = args.patience,
            min_delta = args.min_delta,
            mode = 'min'
            )
        ]
    if args.save_ckpt:
        callbacks.append(
            ModelCheckpoint(
                monitor = 'val_loss',
                dirpath = 'checkpoints',
                save_top_k = 1,
                filename = '{epoch}-{val_loss:.4f}'
                )
            )

    # trainer
    trainer = L.Trainer(
        max_epochs = args.max_epochs,
        logger = logger,
        callbacks = callbacks,
        accelerator = 'auto',
        devices = 'auto',
        num_sanity_val_steps = 0,
        enable_checkpointing = args.save_ckpt,
        gradient_clip_val = args.gradient_clip_val
        )

    # train
    trainer.fit(lit_model, train_dl, val_dl)
