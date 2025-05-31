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
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--hidden_dim', type = int, default = 1024)
    parser.add_argument('--n_layers', type = int, default = 3)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lambda_clf', type = float, default = .3)
    parser.add_argument('--lambda_kldiv', type = float, default = 1e-2)

    # other
    parser.add_argument('--latent_dim', type = int, default = 2)
    parser.add_argument('--max_epochs', type = int, default = 500)
    parser.add_argument('--num_workers', type = int, default = 32)
    args = parser.parse_args()

    # training/validation datasets
    adata = ad.read_h5ad(os.path.join('..', 'data', 'modeling', 'training.h5ad'))    
    train_ix = (adata.obs.training == 'True')
    train_dataset = MesenchymeDataset(adata[train_ix])
    val_dataset = MesenchymeDataset(adata[~train_ix])

    # dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        pin_memory = True
        )
    val_loader = DataLoader(
        val_dataset, 
        batch_size = len(val_dataset),
        shuffle = False,
        num_workers = 1,
        pin_memory = True
        )

    # variational encoder
    model = MesNet(
        input_dim = adata.shape[1],
        target_dim = adata.obsm['X_signature'].shape[1],
        num_classes = adata.obs.celltype.cat.categories.size,
        hidden_dim = args.hidden_dim,
        n_layers = args.n_layers,
        latent_dim = args.latent_dim
        )
    lit_model = MesenchymalStates(args, model)

    # callbacks
    logger = WandbLogger(project = 'mesenchymal-states')
    logger.watch(lit_model, log = 'all', log_freq = 100)
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_loss',
        dirpath = 'checkpoints',
        save_top_k = 1,
        filename = '{epoch}-{val_loss:.4f}'
        )
    earlystop_callback = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        min_delta = 0.,
        mode = 'min'
        )
    callbacks = [checkpoint_callback, earlystop_callback]

    # trainer
    trainer = L.Trainer(
        max_epochs = args.max_epochs,
        logger = logger,
        callbacks = callbacks,
        accelerator = 'auto',
        devices = 'auto',
        log_every_n_steps = 100,
        num_sanity_val_steps = 0
        )

    # train
    trainer.fit(lit_model, train_loader, val_loader)
