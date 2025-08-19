#%%
import os, sys
sys.path.append('..')
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from utils.load import load_CCCA_adata
np.random.seed(1)

pth = os.path.join('..', '..')
pth_out = os.path.join(pth, 'data', 'modeling')

# datasets summary (CCCA)
summary_df = pd.read_csv(os.path.join(pth, 'data', 'CCCA_summary.csv'), sep = '\t')
title = summary_df.Title.str.replace(' et al. ', '')
cat = summary_df.Category.str.replace(' ', '-').replace('/', '-')
summary_df['Name'] = 'Data_' + title + '_' + cat

# dataset features
feat_fn = os.path.join(pth, 'data', 'features', 'biomart', 'union.csv')
feat_union = pd.read_csv(feat_fn)

# preprocessing
def preprocess(adata: ad.AnnData, train_frac: float = .8) -> ad.AnnData:

    # prepare metadata (same as dev dsets)
    if 'cell_type' not in adata.obs.columns:
        adata.obs['cell_type'] = 'Malignant'
    if 'celltype' in adata.obs.columns:
        adata.obs['celltype_orig'] = adata.obs.celltype.copy()
    adata.obs['celltype'] = adata.obs.cell_type.copy()
    adata.obs['trajectory'] = 'False'
    adata.obs['source'] = 'CCCA'

    # training/validation split (celltype stratified)
    celltype_df = adata.obs.groupby('celltype')
    train_ix = celltype_df.sample(frac = train_frac).index
    adata.obs['training'] = adata.obs.index.isin(train_ix).astype(str)

    # keep dset features (mouse genes [homologs])
    feat_msk = adata.var_names.isin(feat_union.hsapiens)
    adata = adata[:, feat_msk].copy()
    var_dict = feat_union.set_index('hsapiens').mmusculus.to_dict()
    adata.var_names = adata.var_names.map(var_dict)
    return adata

# prepare training/validation datasets
adata_dict = dict()
for ix in summary_df.index:
    cat_ix = summary_df.loc[ix, 'Category']
    if cat_ix != 'Other/Models':
        name_ix = summary_df.loc[ix, 'Name']
        dirname = os.path.join(pth, 'data', 'unzip', 'CCCA', name_ix)
        for dirpth, subdir, _ in os.walk(dirname):
            key = name_ix
            if dirpth != dirname:
                key += '_' + os.path.split(dirpth)[1]
            if len(subdir) == 0:
                adata = load_CCCA_adata(dirpth)
                adata = preprocess(adata)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    adata.obs[summary_df.columns] = summary_df.loc[ix]
                adata_dict[key] = adata

# concatenate datasets (all dev dset feaures)
adata = ad.concat(adata_dict, join = 'outer', merge = 'same', label = 'source_CCCA')
adata_dev = sc.read_h5ad(os.path.join(pth_out, 'development.h5ad'))
adata_ccca = ad.AnnData(X = csr_matrix((adata.shape[0], adata_dev.shape[1]), dtype = adata.X.dtype))
adata_ccca.var = adata_dev.var.copy()
adata_ccca.obs = adata.obs.astype(str)
adata_ccca.X = adata[:, adata_ccca.var_names].X
adata_ccca.obs_names_make_unique()

# split malignant/other celltypes
msk_cancer = (adata_ccca.obs.celltype == 'Malignant')
adata_cancer = adata_ccca[msk_cancer].copy()
adata_other = adata_ccca[~msk_cancer].copy()

# split other celltypes in half
msk_half = (np.arange(adata_other.shape[0]) < (adata_other.shape[0] // 2))
adata_other1 = adata_other[msk_half].copy()
adata_other2 = adata_other[~msk_half].copy()

# save datasets
adata_cancer.write(os.path.join(pth_out, 'CCCA_malignant.h5ad'))
adata_other1.write(os.path.join(pth_out, 'CCCA_other1.h5ad'))
adata_other2.write(os.path.join(pth_out, 'CCCA_other2.h5ad'))

#%%
