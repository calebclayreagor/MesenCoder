#%%
import os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
np.random.seed(1)

#%%
summary_df = pd.read_csv(os.path.join('..', 'data', 'summary.csv'), index_col = 0)
features = pd.read_csv(os.path.join('..', 'data', 'features', 'biomart', 'union.csv'))
adata_fn = sorted(glob.glob(os.path.join('..', 'data', 'processed', '*.h5ad')))
adata_dict = {os.path.split(fn)[1].replace('.h5ad', '') : fn for fn in adata_fn}
species_dict = summary_df.Species.to_dict()

#%%
# prepare training/validation datasets
train_frac = .8
for key in adata_dict:
    species = species_dict[key]
    adata = sc.read_h5ad(adata_dict[key])
    adata = adata[~adata.obs.celltype.isna()]

    # keep features from trajectories
    features_msk = adata.var_names.isin(features[species])
    adata = adata[:, features_msk]

    # mouse genes (homologs)
    if species != 'mmusculus':
        features_dict = features.set_index(species).mmusculus.to_dict()
        adata.var_names = adata.var_names.map(features_dict)

    # training/validation split
    train_ix = adata.obs.groupby('celltype').sample(frac = train_frac).index
    adata.obs['training'] = adata.obs.index.isin(train_ix).astype(str)

    adata_dict[key] = adata
    # print(key, adata_dict[key])

#%%
adata = ad.concat(adata_dict, merge = 'same', label = 'Source')
adata.obs_names_make_unique()
adata.write(os.path.join('..', 'data', 'modeling', 'training.h5ad'))

#%%