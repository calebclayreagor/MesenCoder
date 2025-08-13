#%%
import os, sys
import warnings
sys.path.append(os.path.join('..', '..'))
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from utils.load import load_CCCA_adata

# datasets summary
summary_df = pd.read_csv(os.path.join('..', '..', 'data', 'testing.csv'), sep = '\t')
title = summary_df.Title.str.replace(' et al. ', '')
cat = summary_df.Category.str.replace(' ', '-').replace('/', '-')
summary_df['Name'] = 'Data_' + title + '_' + cat

# datasets features
feat_fn = os.path.join('..', '..', 'data', 'features', 'biomart', 'union.csv')
feat_union = pd.read_csv(feat_fn)

# preprocessing
def preprocess(adata: ad.AnnData) -> ad.AnnData:
    # keep dset features (mouse genes [homologs])
    feat_msk = adata.var_names.isin(feat_union.hsapiens)
    if 'cell_type' not in adata.obs.columns:
        adata.obs['cell_type'] = 'Malignant'
    adata = adata[:, feat_msk].copy()
    var_dict = feat_union.set_index('hsapiens').mmusculus.to_dict()
    adata.var_names = adata.var_names.map(var_dict)
    return adata

# prepare testing datasets
adata_dict = dict()
for ix in summary_df.index:
    cat_ix = summary_df.loc[ix, 'Category']
    if cat_ix != 'Other/Models':
        name_ix = summary_df.loc[ix, 'Name']
        dirname = os.path.join('..', '..', 'data', 'unzip', 'CCCA', name_ix)
        for pth, subdir, _ in os.walk(dirname):
            key = name_ix
            if pth != dirname:
                key += '_' + os.path.split(pth)[1]
            if len(subdir) == 0:
                adata = load_CCCA_adata(pth)
                adata = preprocess(adata)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    adata.obs[summary_df.columns] = summary_df.loc[ix]
                adata_dict[key] = adata

# concatenate datasets
adata = ad.concat(adata_dict, join = 'outer', merge = 'same', label = 'source')
adata.obs['weight'] = 1.

# all training features
adata_train = sc.read_h5ad(os.path.join('..', '..', 'data', 'modeling', 'training.h5ad'))
adata_test = ad.AnnData(X = csr_matrix((adata.shape[0], adata_train.shape[1]), dtype = adata.X.dtype))
adata_test.var = adata_train.var.copy()
adata_test.obs = adata.obs.astype(str)
adata_test.obs['celltype'] = adata_test.obs.cell_type
adata_test.X = adata[:, adata_test.var_names].X
adata_test.obs_names_make_unique()

# split malignant & other cells (x2)
msk_malignant = (adata_test.obs.celltype == 'Malignant')
adata_malignant = adata_test[msk_malignant].copy()
adata_other = adata_test[~msk_malignant].copy()
msk_other1 = (np.arange(adata_other.shape[0]) < (adata_other.shape[0] // 2))
adata_other1 = adata_other[msk_other1].copy()
adata_other2 = adata_other[~msk_other1].copy()

# save datasets
pth_out = os.path.join('..', '..', 'data', 'modeling', 'testing', 'CCCA')
adata_malignant.write(os.path.join(pth_out, 'malignant.h5ad'))
adata_other1.write(os.path.join(pth_out, 'other_1.h5ad'))
adata_other2.write(os.path.join(pth_out, 'other_2.h5ad'))

#%%
