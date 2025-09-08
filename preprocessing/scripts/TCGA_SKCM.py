#%%
import os
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

pth = os.path.join('..', '..')
pth_in = os.path.join(pth, 'data', 'unzip', 'TCGA-SKCM')
pth_feat = os.path.join(pth, 'data', 'features', 'biomart')
pth_out = os.path.join(pth, 'data', 'modeling')

# load dataset [log2(1 + avg_TPM) -> log1p(avg_TPM)]
fn_data = os.path.join(pth_in, 'TCGA-SKCM.star_tpm.tsv')
adata = sc.read_csv(fn_data, delimiter = '\t').T
adata.X = csr_matrix(2 ** adata.X - 1)
sc.pp.log1p(adata)

# map IDs
fn_map = os.path.join(pth_in, 'gencode.v36.annotation.gtf.gene.probemap.tsv')
id_dict = pd.read_csv(fn_map, delimiter = '\t', index_col = 0).gene.to_dict()
adata.var_names = adata.var_names.map(id_dict)

# load metadata (clinical, survival)
fn_clinical = os.path.join(pth_in, 'TCGA-SKCM.clinical.tsv')
fn_survival = os.path.join(pth_in, 'TCGA-SKCM.survival.tsv')
df_clinical = pd.read_csv(fn_clinical, delimiter = '\t', index_col = 0)
df_survival = pd.read_csv(fn_survival, delimiter = '\t', index_col = 0)
df_obs = pd.concat((df_clinical, df_survival), axis = 1)
df_obs['source'] = 'CCCA'
adata.obs = adata.obs.join(df_obs.astype(str), how = 'left')

# keep features (mouse genes [homologs])
features = pd.read_csv(os.path.join(pth_feat, 'union.csv'))
feat_msk = adata.var_names.isin(features.hsapiens)
adata = adata[:, feat_msk].copy()
var_dict = features.set_index('hsapiens').mmusculus.to_dict()
adata.var_names = adata.var_names.map(var_dict)

# all training features
adata_dev = sc.read_h5ad(os.path.join(pth_out, 'development.h5ad'))
X_skcm = csr_matrix((adata.shape[0], adata_dev.shape[1]), dtype = adata.X.dtype)
X_skcm[:, adata_dev.var_names.get_indexer(adata.var_names)] = adata.X.copy()
adata_skcm = ad.AnnData(X = X_skcm, var = adata_dev.var, obs = adata.obs)
adata_skcm.write(os.path.join(pth_out, 'TCGA_SKCM.h5ad'))

#%%
