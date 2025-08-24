import argparse
import lightning as L
import anndata as ad
from torch.utils.data import DataLoader
from lit_module import MesenchymalStates
from dataset import MesenchymeDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # CLI parameters
    parser.add_argument('--adata_pth', type = str)
    parser.add_argument('--out_pth', type = str)
    parser.add_argument('--ckpt_pth', type = str)
    parser.add_argument('--batch_size', type = int, default = 1024)
    parser.add_argument('--num_workers', type = int, default = 32)
    args = parser.parse_args()

    L.seed_everything(1)

    # prediction dataset
    adata = ad.read_h5ad(args.adata_pth)
    pred_ds = MesenchymeDataset(adata)

    # dataloader
    pred_dl = DataLoader(
        pred_ds,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True)

    # trained autoencoder
    lit_model = MesenchymalStates.load_from_checkpoint(
        args.ckpt_pth, out_pth = args.out_pth)

    # predict
    trainer = L.Trainer()
    trainer.predict(lit_model, pred_dl)
