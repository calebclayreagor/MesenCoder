import os, argparse
import lightning as L
import anndata as ad
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lit_module import MesenchymalStates
from dataset import MesenchymeDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # CLI parameters
    parser.add_argument('--hidden_dim', type = int, default = 32)
    parser.add_argument('--latent_dim_src', type = int, default = 4)
    parser.add_argument('--batch_size', type = int, default = 1024)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)
    parser.add_argument('--max_epochs', type = int, default = 200)
    parser.add_argument('--val_log_freq', type = int, default = 10)
    parser.add_argument('--save_ckpt', action = 'store_true')
    parser.add_argument('--wandb_project', type = str, default = 'MesenCoder')
    parser.add_argument('--num_workers', type = int, default = 32)
    args = parser.parse_args()
    L.seed_everything(1)

    # training/validation datasets
    datadir = os.path.join('..', 'data', 'modeling', 'inputs')
    adata = ad.read_h5ad(os.path.join(datadir, 'development.h5ad'))
    adata_train = adata[adata.obs.training == 'True'].copy()
    adata_val = adata[adata.obs.validation == 'True'].copy()
    train_ds = MesenchymeDataset(adata_train, True)
    val_ds = MesenchymeDataset(adata_val, True)

    # training sampler
    sampler = WeightedRandomSampler(
        weights = adata_train.obs.weight.values,
        num_samples = len(train_ds),
        replacement = True)

    # dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        sampler = sampler,
        num_workers = args.num_workers,
        pin_memory = True)
    val_dl = DataLoader(
        val_ds,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True)

    # custom autoencoder
    args.n_feature = adata.shape[1]
    args.n_source = adata.obs.source.cat.categories.nunique()
    model = MesenchymalStates(args)

    # wandb logger
    logger = WandbLogger(
        project = args.wandb_project,
        log_model = args.save_ckpt)
    logger.watch(model, log = 'all')

    # checkpoints
    if args.save_ckpt:
        checkpoints = ModelCheckpoint(
            every_n_epochs = args.val_log_freq,
            save_top_k = -1,
            save_last = False,
            dirpath = os.path.join(
                'checkpoints', args.wandb_project),
            filename = '{epoch:02d}')
    else: checkpoints = None

    # trainer
    trainer = L.Trainer(
        max_epochs = args.max_epochs,
        logger = logger,
        callbacks = checkpoints,
        accelerator = 'auto',
        devices = 'auto',
        num_sanity_val_steps = 0,
        log_every_n_steps = 4,
        enable_checkpointing = args.save_ckpt)

    # train
    trainer.fit(model, train_dl, val_dl)
