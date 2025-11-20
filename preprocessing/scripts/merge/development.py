#%%
import os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.stats import gmean
np.random.seed(1)

pth = os.path.join('..', '..', '..', 'data')
pth_out = os.path.join(pth, 'modeling', 'inputs')

# training/validation split
train_frac = .8
n_cells_min = lambda g: len(g) > 200

# datasets summary
summary_df = pd.read_csv(os.path.join(pth, 'summary.csv'), index_col = 0)
summary_df['Training'] = summary_df.Training.astype(bool)
species_dict = summary_df.Species.to_dict()

# processed datasets (developmental)
adata_fn = sorted(glob.glob(os.path.join(pth, 'processed', '*.h5ad')))
adata_dict = {os.path.split(fn)[1].replace('.h5ad', '') : fn for fn in adata_fn}

# pseudotime trajectories
traj_fn = sorted(glob.glob(os.path.join(pth, 'trajectories', '*.h5ad')))
traj_dict = {os.path.split(fn)[1].replace('.h5ad', '') : fn for fn in traj_fn}

# dataset features
feat_fn = os.path.join(pth, 'features', 'biomart', 'union.csv')
feat_union = pd.read_csv(feat_fn)

# prepare training/validation datasets
for key in adata_dict:
    df_key = summary_df.loc[key]
    species = species_dict[key]
    adata = sc.read_h5ad(adata_dict[key])
    traj = sc.read_h5ad(traj_dict[key])

    # keep features
    adata = adata[:, adata.var_names.isin(feat_union[species])].copy()

    # annotate trajectory (cells/embedding/t)
    msk_traj = adata.obs_names.isin(traj.obs_names)
    traj_names = adata.obs_names[msk_traj]
    X_traj_name = df_key.loc['Trajectory Embedding']
    X_traj = np.full((adata.shape[0], 2), np.nan)
    X_traj[msk_traj] = traj[traj_names].obsm[X_traj_name]
    t = traj[traj_names].obs.t.copy()
    adata.obsm['X_trajectory'] = X_traj
    adata.obs['trajectory'] = msk_traj
    adata.obs.loc[traj_names, 't'] = t

    # mouse genes (homologs)
    if species != 'mmusculus':
        var_dict = feat_union.set_index(species).mmusculus.to_dict()
        adata.var_names = adata.var_names.map(var_dict)

    # training/validation split — trajectory only (celltype stratified)
    if df_key.Training:
        celltype_df = adata[msk_traj].obs.groupby('celltype').filter(n_cells_min)
        train_ix = celltype_df.sample(frac = train_frac).index
        adata.obs['training'] = adata.obs_names.isin(train_ix)
        adata.obs['validation'] = adata.obs.trajectory & ~adata.obs.training
    else: adata.obs[['training', 'validation']] = False
    adata_dict[key] = adata

# concatenate datasets
adata = ad.concat(adata_dict,
                  join = 'outer',
                  merge = 'same',
                  label = 'source')
adata.obs_names_make_unique()
adata.obs = adata.obs.astype(str)

# weights (celltype/source)
for col in ('celltype', 'source'):
    col_size = adata.obs.groupby(col).size()
    col_weight = 1 / adata.obs[col].map(col_size)
    adata.obs[f'weight_{col}'] = col_weight
weight = adata.obs[['weight_celltype', 'weight_source']]
weight = gmean(weight.values, axis = 1)
weight /= weight.mean()
adata.obs['weight'] = weight

# save dataset
adata.write(os.path.join(pth_out, 'development.h5ad'))

#%%
