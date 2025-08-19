#%%
import os
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.stats import gmean

# load datasets
pth = os.path.join('..', '..', 'data', 'modeling')
adata_dev = sc.read_h5ad(os.path.join(pth, 'development.h5ad'))
adata_cancer = sc.read_h5ad(os.path.join(pth, 'CCCA_malignant.h5ad'))
adata = ad.concat((adata_dev, adata_cancer), join = 'outer', merge = 'same')
adata.obs = adata.obs.astype(str)
adata.obs_names_make_unique()

# category (celltype/disease)
cat = adata.obs.celltype.copy()
msk_cancer = (cat == 'Malignant')
cat_cancer = adata[msk_cancer].obs.Disease.copy()
cat.loc[msk_cancer] = cat_cancer
adata.obs['category'] = cat.copy()

# weights (category/dataset)
for col in ('category', 'source'):
    col_size = adata.obs.groupby(col).size()
    col_weight = 1 / adata.obs[col].map(col_size)
    adata.obs[f'weight_{col}'] = col_weight
weight = adata.obs[['weight_category', 'weight_source']]
weight = gmean(weight.values, axis = 1)
weight /= weight.mean()
adata.obs['weight'] = np.clip(weight, a_min = None, a_max = 10.)

# save dataset
adata.write(os.path.join(pth, 'training.h5ad'))

#%%
