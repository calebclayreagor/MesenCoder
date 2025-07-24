#%%
import os, sys
import warnings
sys.path.append('..')
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from utils.load import load_CCCA_adata

# datasets summary
summary_df = pd.read_csv(os.path.join('..', 'data', 'testing.csv'), sep = '\t')
title = summary_df.Title.str.replace(' et al. ', '')
cat = summary_df.Category.str.replace(' ', '-').replace('/', '-')
summary_df['Name'] = 'Data_' + title + '_' + cat

# datasets features
feat_fn = os.path.join('..', 'data', 'features', 'biomart', 'union.csv')
feat_union = pd.read_csv(feat_fn)

# preprocessing
def preprocess(adata: ad.AnnData) -> ad.AnnData:
    # keep malignant cells, dset features (mouse genes [homologs])
    feat_msk = adata.var_names.isin(feat_union.hsapiens)
    if 'cell_type' not in adata.obs.columns:
        adata.obs['cell_type'] = 'Malignant'
    adata = adata[adata.obs.cell_type == 'Malignant', feat_msk].copy()
    var_dict = feat_union.set_index('hsapiens').mmusculus.to_dict()
    adata.var_names = adata.var_names.map(var_dict)
    return adata

# prepare testing datasets
adata_dict = dict()
for ix in summary_df.index:
    cat_ix = summary_df.loc[ix, 'Category']
    if cat_ix != 'Other/Models':
        name_ix = summary_df.loc[ix, 'Name']
        dirname = os.path.join('..', 'data', 'unzip', 'CCCA', name_ix)
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
adata_train = sc.read_h5ad(os.path.join('..', 'data', 'modeling', 'training.h5ad'))
adata_test = ad.AnnData(X = csr_matrix((adata.shape[0], adata_train.shape[1]), dtype = adata.X.dtype))
adata_test.var = adata_train.var.copy()
adata_test.obs = adata.obs.copy()
adata_test.obs['celltype'] = adata_test.obs.cell_type
adata_test[:, adata.var_names].X = adata.X
adata_test.obs_names_make_unique()

# save dataset
adata_test.write(os.path.join('..', 'data', 'modeling', 'testing.h5ad'))

#%%
