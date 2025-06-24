import torch
from torch.utils.data import Dataset
import numpy as np

class MesenchymeDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata
        self.X = adata.X.toarray().astype(np.float32)
        self.src = adata.obs.source.cat.codes.astype(np.int64).values
        self.ct = adata.obs.celltype.cat.codes.astype(np.int64).values
        self.w = adata.obs.weight.astype(np.float32).values

        # as tensors
        self.X = torch.tensor(self.X, dtype = torch.float32)
        self.src = torch.tensor(self.src, dtype = torch.int64)
        self.ct = torch.tensor(self.ct, dtype = torch.int64)
        self.w = torch.tensor(self.w, dtype = torch.float32)

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, i):
        X1 = self.X[i]
        src1 = self.src[i]
        ct1 = self.ct[i]
        w1 = self.w[i]

        # contrastive example
        ct_msk = (self.ct == ct1); ct_msk[i] = False
        ix_j = ct_msk.nonzero(as_tuple = True)[0]
        j = ix_j[torch.randint(len(ix_j), (1,))].item()
        X2 = self.X[j]
        src2 = self.src[j]
        w2 = self.w[j]

        return (X1, src1, w1), (X2, src2, w2)
