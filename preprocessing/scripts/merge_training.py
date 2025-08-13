#%%
import os
import scanpy as sc
import anndata as ad

# load datasets
pth = os.path.join('..', '..', 'data', 'modeling')
adata_dev = sc.read_h5ad(os.path.join(pth, 'development.h5ad'))
adata_cancer = sc.read_h5ad(os.path.join(pth, 'CCCA_malignant.h5ad'))
adata = ad.concat((adata_dev, adata_cancer), join = 'outer', merge = 'same')
adata.obs = adata.obs.astype(str)
adata.obs_names_make_unique()

# dataset weights
weight = adata.shape[0] / adata.obs.groupby('source').size()
weight /= weight.mean()
adata.obs['weight'] = adata.obs.source.map(weight)

# save dataset
adata.write(os.path.join(pth, 'training.h5ad'))

#%%
