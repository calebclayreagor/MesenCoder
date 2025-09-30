import argparse
import torch
import numpy as np
import lightning as L
import anndata as ad
from tqdm import tqdm
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
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
    parser.add_argument('--device', type = str, default = 'cuda')
    args = parser.parse_args()

    L.seed_everything(1)
    device = torch.device(args.device)

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
        args.ckpt_pth, out_pth = '')

    # integrated gradients w.r.t. latent_z
    model = lit_model.model.to(device).eval()
    ig = IntegratedGradients(lambda x, src: model(x, src)[1])

    # prediction loop
    attr = np.zeros(adata.shape)
    for batch in tqdm(pred_dl):
        X, src, ix = batch
        X, src = X.to(device), src.to(device)
        attr_ix = ig.attribute(inputs = X,
            additional_forward_args = src)
        attr[ix] = attr_ix.detach().cpu().numpy()

    # save output
    attr = csr_matrix(attr)
    adata.layers['integrated_gradients'] = attr
    adata.write(args.out_pth)