#%%
import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

pth = os.path.join('..', 'data')
pth_in = os.path.join(pth, 'unzip', 'TCGA')
pth_out = os.path.join(pth, 'TCGA')

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
adata.var_names_make_unique()

# load phenotype data
fn_pheno = os.path.join(pth_in, 'Survival_SupplementalTable_S1_20171025_xena_sp.tsv')
df = pd.read_csv(fn_pheno, delimiter = '\t', index_col = 0)
adata.obs = adata.obs.join(df.astype(str), how = 'left')

# save output
adata.write(os.path.join(pth_out, 'PANCAN.h5ad'))

#%%
