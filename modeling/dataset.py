import torch
from torch.utils.data import Dataset
import numpy as np
import anndata as ad

class MesenchymeDataset(Dataset):
    def __init__(self, adata: ad.AnnData) -> None:
        self.adata = adata
        self.X = adata.X.toarray().astype(np.float32)
        self.src = adata.obs.source.cat.codes.astype(np.int64).values
        self.w = adata.obs.weight.astype(np.float32).values

        # as tensors
        self.X = torch.tensor(self.X, dtype = torch.float32)
        self.src = torch.tensor(self.src, dtype = torch.int64)
        self.w = torch.tensor(self.w, dtype = torch.float32)

    def __len__(self) -> int:
        return self.adata.shape[0]

    def __getitem__(self, i: int
                    ) -> tuple[torch.Tensor,
                               torch.Tensor,
                               torch.Tensor]:
        X = self.X[i]
        src = self.src[i]
        w = self.w[i]
        return X, src, w

