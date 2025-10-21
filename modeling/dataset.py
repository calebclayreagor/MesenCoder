import torch
from torch.utils.data import Dataset
import numpy as np
import anndata as ad

class MesenchymeDataset(Dataset):
    def __init__(self, adata: ad.AnnData, decode: bool) -> None:
        self.adata = adata
        self.decode = decode
        self.X = adata.X.toarray().astype(np.float32)
        self.ix = np.arange(adata.n_obs).astype(np.int64)
        if self.decode:
            self.src = adata.obs.source.cat.codes.astype(np.int64).values

        # as tensors
        self.X = torch.tensor(self.X, dtype = torch.float32)
        self.ix = torch.tensor(self.ix, dtype = torch.int64)
        if self.decode:
            self.src = torch.tensor(self.src, dtype = torch.int64)

    def __len__(self) -> int:
        return self.adata.shape[0]

    def __getitem__(self, i: int
                    ) -> tuple[torch.Tensor,
                               torch.Tensor] | \
                         tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor]:
        X = self.X[i]
        ix = self.ix[i]
        if self.decode:
            src = self.src[i]
            return X, src, ix
        else:
            return X, ix

