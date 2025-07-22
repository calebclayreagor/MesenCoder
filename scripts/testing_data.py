#%%
import os, glob
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

# datasets summary
summary_df = pd.read_csv(os.path.join('..', 'data', 'summary.csv'), index_col = 0)
summary_df = summary_df.loc[summary_df.Split == 'Testing']
species_dict = summary_df.Species.to_dict()

# processed datasets
adata_fn = sorted(glob.glob(os.path.join('..', 'data', 'processed', 'Testing', '*', '*.h5ad')))
adata_dict = {os.path.split(fn)[1].replace('.h5ad', '') : fn for fn in adata_fn}
adata_training = sc.read_h5ad(os.path.join('..', 'data', 'modeling', 'training.h5ad'))

# datasets features
feat_fn = os.path.join('..', 'data', 'features', 'biomart', 'union.csv')
feat_union = pd.read_csv(feat_fn)

# prepare testing datasets
for key in adata_dict:
    species = species_dict[key]
    adata = sc.read_h5ad(adata_dict[key])

    # keep malignant cells, features
    features_msk = adata.var_names.isin(feat_union[species])
    adata = adata[adata.obs.celltype == 'Malignant', features_msk]

    # mouse genes (homologs)
    if species != 'mmusculus':
        var_dict = feat_union.set_index(species).mmusculus.to_dict()
        adata.var_names = adata.var_names.map(var_dict)
    adata_dict[key] = adata

# concatenate datasets
adata = ad.concat(adata_dict, join = 'outer', merge = 'same', label = 'source')
adata.obs['weight'] = 1.

# training features
adata_testing = ad.AnnData(X = csr_matrix((adata.shape[0], adata_training.shape[1]), dtype = adata.X.dtype))
adata_testing.var = adata_training.var.copy()
adata_testing.obs = adata.obs.copy()
adata_testing[:, adata.var_names].X = adata.X

# save dataset
adata_testing.write(os.path.join('..', 'data', 'modeling', 'testing.h5ad'))

#%%
