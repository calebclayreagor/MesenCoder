#%%
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'unzip', 'TCGA')
pth_feat = os.path.join(pth, 'features', 'biomart')
pth_out = os.path.join(pth, 'modeling', 'inputs')

# load dataset [log2(TPM + 1e-3) -> log1p(TPM)]
fn_data = os.path.join(pth_in, 'tcga_RSEM_gene_tpm.tsv')
adata = sc.read_csv(fn_data, delimiter = '\t').T
adata.X = np.exp2(adata.X) - 1e-3
adata.X = np.clip(adata.X, 0., None)
adata.X = csr_matrix(adata.X)
sc.pp.log1p(adata)

# map IDs
fn_map = os.path.join(pth_in, 'gencode.v23.annotation.gene.probemap.tsv')
id_dict = pd.read_csv(fn_map, delimiter = '\t', index_col = 0).gene.to_dict()
adata.var_names = adata.var_names.map(id_dict)

# load phenotype data
fn_pheno = os.path.join(pth_in, 'Survival_SupplementalTable_S1_20171025_xena_sp.tsv')
df = pd.read_csv(fn_pheno, delimiter = '\t', index_col = 0)
adata.obs = adata.obs.join(df.astype(str), how = 'left')

# keep features (mouse genes [homologs])
features = pd.read_csv(os.path.join(pth_feat, 'union.csv'))
adata = adata[:, adata.var_names.isin(features.hsapiens)].copy()
var_dict = features.set_index('hsapiens').mmusculus.to_dict()
adata.var_names = adata.var_names.map(var_dict)

# all dev features
adata_dev = sc.read_h5ad(os.path.join(pth_out, 'development.h5ad'))
X = csr_matrix((adata.shape[0], adata_dev.shape[1]), dtype = adata.X.dtype)
X[:, adata_dev.var_names.get_indexer(adata.var_names)] = adata.X.copy()
adata_out = ad.AnnData(X = X, var = adata_dev.var, obs = adata.obs)
adata_out.write(os.path.join(pth_out, 'TCGA.h5ad'))

#%%
