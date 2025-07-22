#%%
import os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

adata_training = sc.read_h5ad(os.path.join('..', 'data', 'modeling', 'training.h5ad'))
adata_testing = sc.read_h5ad(os.path.join('..', 'data', 'processed', 'CCCA_Neuroendocrine.h5ad'))

feat_fn = os.path.join('..', 'data', 'features', 'biomart', 'union.csv')
feat_union = pd.read_csv(feat_fn)

#%%
# keep features
features_msk = adata_testing.var_names.isin(feat_union['hsapiens'])
adata_testing = adata_testing[~adata_testing.obs.celltype.isna(), features_msk]

#%%
# mouse genes (homologs)
var_dict = feat_union.set_index('hsapiens').mmusculus.to_dict()
adata_testing.var_names = adata_testing.var_names.map(var_dict)

#%%
# Create new matrix with zeros, same shape as (n_obs, n_genes in reference)
X_new = np.zeros((adata_testing.n_obs, adata_training.n_vars))

# Fill in values for shared genes
shared_genes = adata_training.var_names.intersection(adata_testing.var_names)
X_new[:, [adata_training.var_names.get_loc(g) for g in shared_genes]] = adata_testing[:, shared_genes].X.toarray()

# Rebuild adata with aligned genes
adata_testing = ad.AnnData(X = X_new, obs = adata_testing.obs.copy(), var = adata_training.var.copy())
adata_testing.var_names = adata_training.var_names

#%%
from scipy.sparse import csr_matrix
adata_testing.X = csr_matrix(adata_testing.X)

#%%
adata_testing.write(os.path.join('..', 'data', 'modeling', 'testing.h5ad'))

#%%
