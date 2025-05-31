#%%
import os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
np.random.seed(1)

# datasets summary
summary_df = pd.read_csv(os.path.join('..', 'data', 'summary.csv'), index_col = 0)
species_dict = summary_df.Species.to_dict()

# processed datasets
adata_fn = sorted(glob.glob(os.path.join('..', 'data', 'processed', '*.h5ad')))
adata_dict = {os.path.split(fn)[1].replace('.h5ad', '') : fn for fn in adata_fn}

# trajectory features
feat_fn = sorted(glob.glob(os.path.join('..', 'data', 'features', 'biomart', '*.csv')))
feat_dict = {os.path.split(fn)[1].replace('.csv', '') : pd.read_csv(fn) for fn in feat_fn}
feat_union = feat_dict['union']; del feat_dict['union']

#%%
# prepare training/validation datasets
train_frac = .8
for key in adata_dict:
    species = species_dict[key]
    adata = sc.read_h5ad(adata_dict[key])

    # keep celltypes, features
    features_msk = adata.var_names.isin(feat_union[species])
    adata = adata[~adata.obs.celltype.isna(), features_msk]

    # trajectory signatures (scaled)
    adata.obsm['X_signature'] = np.zeros((adata.shape[0], len(feat_dict)))
    for i, (source, df) in enumerate(feat_dict.items()):
        sc.tl.score_genes(adata, df[species], score_name = f'{source}_signature', ctrl_as_ref = True)
        adata.obsm['X_signature'][:, i] = adata.obs[f'{source}_signature'].values
    adata.obsm['X_signature'] = StandardScaler().fit_transform(adata.obsm['X_signature'])

    # scaled features (sparse)
    sc.pp.scale(adata)
    adata.X = csr_matrix(adata.X)

    # mouse genes (homologs)
    if species != 'mmusculus':
        var_dict = feat_union.set_index(species).mmusculus.to_dict()
        adata.var_names = adata.var_names.map(var_dict)

    # training/validation split
    celltype_df = adata.obs.groupby('celltype')
    train_ix = celltype_df.sample(frac = train_frac).index
    adata.obs['training'] = adata.obs.index.isin(train_ix).astype(str)

    # num celltype (weights)
    adata.obs['n_celltype'] = adata.obs.celltype.map(celltype_df.size().to_dict()).astype(int)

    adata_dict[key] = adata
    # print(key, adata_dict[key])

#%%
adata = ad.concat(adata_dict, join = 'outer', merge = 'same', label = 'Source')
adata.obs_names_make_unique()
adata.obs['weight'] = adata.shape[0] / adata.obs.n_celltype
adata.obs['weight'] = adata.obs.weight / adata.obs.weight.mean()
adata.write(os.path.join('..', 'data', 'modeling', 'training.h5ad'))

#%%
