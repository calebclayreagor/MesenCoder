import torch
from torch.utils.data import Dataset
import numpy as np

class MesenchymeDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata
        self.X = adata.X.toarray()
        self.y = adata.obsm['X_signature'].astype(np.float32)
        self.c = adata.obs.celltype.cat.codes.values.astype(np.int64)
        self.w = adata.obs.weight.values.astype(np.float32)

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, i):
        X = torch.tensor(self.X[i], dtype = torch.float32)
        y = torch.tensor(self.y[i], dtype = torch.float32)
        c = torch.tensor(self.c[i], dtype = torch.long)
        w = torch.tensor(self.w[i], dtype = torch.float32)
        return X, y, c, w
