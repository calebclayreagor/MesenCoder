#%%
import os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
np.random.seed(1)

# training/validation split
train_frac = .8

# datasets summary
summary_df = pd.read_csv(os.path.join('..', 'data', 'training.csv'), index_col = 0)
species_dict = summary_df.Species.to_dict()

# processed datasets
adata_fn = sorted(glob.glob(os.path.join('..', 'data', 'processed', '*.h5ad')))
adata_dict = {os.path.split(fn)[1].replace('.h5ad', '') : fn for fn in adata_fn}

# pseudotime trajectories
traj_fn = sorted(glob.glob(os.path.join('..', 'data', 'trajectories', '*.h5ad')))
traj_dict = {os.path.split(fn)[1].replace('.h5ad', '') : fn for fn in traj_fn}

# trajectory features
feat_fn = os.path.join('..', 'data', 'features', 'biomart', 'union.csv')
feat_union = pd.read_csv(feat_fn)

# prepare training/validation datasets
for key in adata_dict:
    species = species_dict[key]
    adata = sc.read_h5ad(adata_dict[key])
    traj = sc.read_h5ad(traj_dict[key])

    # keep celltypes, features
    features_msk = adata.var_names.isin(feat_union[species])
    adata = adata[~adata.obs.celltype.isna(), features_msk]

    # trajectory cells
    adata.obs['trajectory'] = adata.obs_names.isin(traj.obs_names)

    # mouse genes (homologs)
    if species != 'mmusculus':
        var_dict = feat_union.set_index(species).mmusculus.to_dict()
        adata.var_names = adata.var_names.map(var_dict)

    # training/validation split (celltype stratify)
    celltype_df = adata.obs.groupby('celltype')
    train_ix = celltype_df.sample(frac = train_frac).index
    adata.obs['training'] = adata.obs.index.isin(train_ix).astype(str)
    adata_dict[key] = adata

# concatenate datasets
adata = ad.concat(adata_dict, join = 'outer', merge = 'same', label = 'source')
adata.obs_names_make_unique()

# dataset weights
weight = adata.shape[0] / adata.obs.groupby('source').size()
weight /= weight.mean()
adata.obs['weight'] = adata.obs.source.map(weight)

# save dataset
adata.write(os.path.join('..', 'data', 'modeling', 'training.h5ad'))

#%%
