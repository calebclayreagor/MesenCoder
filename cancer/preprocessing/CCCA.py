#%%
import os
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from utils.load import load_CCCA_adata

pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'unzip', 'CCCA')
pth_feat = os.path.join(pth, 'features', 'biomart')
pth_out = os.path.join(pth, 'modeling', 'inputs')

# datasets summary (CCCA)
summary_df = pd.read_csv(os.path.join(pth, 'CCCA_summary.csv'), sep = '\t')
title = summary_df.Title.str.replace(' et al. ', '')
cat = summary_df.Category.str.replace(r'[ /]', '-', regex = True)
summary_df['Name'] = 'Data_' + title + '_' + cat

# dataset features
feat_union = pd.read_csv(os.path.join(pth_feat, 'union.csv'))
var_dict = feat_union.set_index('hsapiens').mmusculus.to_dict()

# preprocessing
def preprocess(adata: ad.AnnData) -> ad.AnnData:
    if 'cell_type' not in adata.obs.columns:
        adata.obs['cell_type'] = 'Malignant'
    if 'celltype' in adata.obs.columns:
        adata.obs['celltype_orig'] = adata.obs.celltype.copy()
    adata.obs['celltype'] = adata.obs.cell_type.copy()

    # keep dset features (mouse genes [homologs])
    adata = adata[:, adata.var_names.isin(feat_union.hsapiens)].copy()
    adata.var_names = adata.var_names.map(var_dict)
    return adata

# prepare datasets
adata_dict = dict()
for ix in summary_df.index:
    name_ix = summary_df.loc[ix].Name
    dirname = os.path.join(pth_in, name_ix)
    if os.path.exists(dirname):
        for dirpth, subdir, _ in os.walk(dirname):
            if len(subdir) == 0:
                adata = load_CCCA_adata(dirpth)
                adata = preprocess(adata)
                for col in summary_df.columns:
                    adata.obs[col] = summary_df.loc[ix, col]
                adata_dict[dirpth] = adata
    else:
        print(f'WARNING: The directory {dirname} does not exist!')

# concatenate datasets (all dev feaures)
adata = ad.concat(adata_dict,
                  join = 'outer',
                  merge = 'same',
                  label = 'source')
fn = os.path.join(pth_out, 'development.h5ad')
adata_dev = sc.read_h5ad(fn)
X = csr_matrix((adata.shape[0], adata_dev.shape[1]),
               dtype = adata.X.dtype)
adata_ccca = ad.AnnData(X = X)
adata_ccca.var = adata_dev.var.copy()
adata_ccca.obs = adata.obs.astype(str)
adata_ccca.X = adata[:, adata_ccca.var_names].X
adata_ccca.obs_names_make_unique()

# save malignant cells
msk_cancer = (adata_ccca.obs.celltype == 'Malignant')
adata_cancer = adata_ccca[msk_cancer].copy()
adata_cancer.write(os.path.join(pth_out, 'CCCA.h5ad'))

#%%
