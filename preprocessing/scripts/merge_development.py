#%%
import os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
np.random.seed(1)
pth = os.path.join('..', '..', 'data')

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

    # trajectory (cells/embedding)
    msk_traj = adata.obs_names.isin(traj.obs_names)
    traj_names = adata.obs_names[msk_traj]
    X_traj_name = df_key.loc['Trajectory Embedding']
    X_traj = np.zeros((adata.shape[0], 2))
    X_traj[msk_traj] = traj[traj_names].obsm[X_traj_name]
    adata.obsm['X_trajectory'] = X_traj
    adata.obs['trajectory'] = msk_traj

    # mouse genes (homologs)
    if species != 'mmusculus':
        var_dict = feat_union.set_index(species).mmusculus.to_dict()
        adata.var_names = adata.var_names.map(var_dict)

    # training/validation split — trajectory only (celltype stratified)
    if df_key.Training:
        celltype_df = adata[msk_traj].obs.groupby('celltype').filter(n_cells_min)
        train_ix = celltype_df.sample(frac = train_frac).index
        adata.obs['training'] = adata.obs_names.isin(train_ix).astype(str)
    else: adata.obs['training'] = 'False'
    adata_dict[key] = adata

# concatenate datasets
adata = ad.concat(adata_dict, join = 'outer', merge = 'same', label = 'source')
adata.obs_names_make_unique()

# save dataset
adata.write(os.path.join(pth, 'modeling', 'development.h5ad'))

#%%
