import argparse
import torch
import numpy as np
import lightning as L
import anndata as ad
from tqdm import tqdm
import torch.nn.functional as F
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
    model = lit_model.model.to(device).eval()
    hidden_dim = model.decoder[0].out_features
    W_lin = model.decoder[-1].tied_to.weight

    # IG_x(z) (encoder attribution)
    def fwd_enc(x: torch.Tensor) -> torch.Tensor:
        u = model.encoder(x)
        return torch.log1p(F.softplus(u) + 1e-6)
    ig_z = IntegratedGradients(fwd_enc)

    # IG_v(x_hat) (decoder attribution)
    def fwd_dec(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = torch.cat((z, v), dim = -1)
        return model.decoder[:-1](h)  # return hidden_out
    ig_v = IntegratedGradients(fwd_dec)

    # loop over mini-batches
    attr_z = np.zeros(adata.shape)
    attr_v = np.zeros(adata.shape)
    for batch in tqdm(pred_dl):
        X, src, ix = batch
        X, src = X.to(device), src.to(device)

        # IG_x(z)
        attr_z[ix] = ig_z.attribute(inputs = X).detach().cpu().numpy()

        # encoder (no grad)
        with torch.no_grad():
            z = fwd_enc(X)
            v = model.embed_src(src)

        # IG_v(hidden_out)
        attr_v_ix = torch.zeros((X.size(0), hidden_dim), device = device)
        for jx in range(hidden_dim):
            attr_v_ix[:, jx] = ig_v.attribute(
                inputs = v, additional_forward_args = z, target = jx).sum(-1)

        # IG_v(x_hat)
        attr_v[ix] = (attr_v_ix @ W_lin).detach().cpu().numpy()

    # save output
    adata.layers['IG_z'] = csr_matrix(attr_z)
    adata.layers['IG_v'] = csr_matrix(attr_v)
    adata.write(args.out_pth)