#%%
import os, glob
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm

# params
name, groupby = 'CCCA', 'source'

# load data
pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'modeling', 'inputs')
pth_feat = os.path.join(pth, 'features', 'biomart')
pth_out = os.path.join(pth, 'modeling', 'predict')
adata = sc.read_h5ad(os.path.join(pth_in, f'{name}.h5ad'))
reg = pd.read_csv('betareg.csv', index_col = 0)

# module signatures
feat_fn = sorted(glob.glob(os.path.join(pth_feat, '*.csv')))
df = pd.read_csv(os.path.join(pth, 'summary.csv'), index_col = 0)
get_source = lambda fn: os.path.split(fn)[1].replace('.csv', '')
feat_dict = {get_source(fn) : fn for fn in feat_fn if 'union' not in fn}
for src, fn in feat_dict.items():
    print(src)
    df_feat = pd.read_csv(fn)
    g = df_feat.mmusculus.copy()
    g = g.loc[g.isin(adata.var_names)]
    for _, df_group in adata.obs.groupby(groupby, observed = True):
        msk = adata.obs_names.isin(df_group.index)
        adata_group = adata[msk].copy()
        sc.tl.score_genes(adata_group,
                          gene_list = g,
                          ctrl_as_ref = False,
                          score_name = src,
                          random_state = 1234)
        adata.obs.loc[msk, src] = adata_group.obs[src]

# whiten features
src_train = df.loc[df.Training == True].index
adata.obs[src_train] = (adata.obs[src_train]
                        .sub(reg.loc[src_train].mu, axis = 1)
                        .div(reg.loc[src_train].sigma, axis = 1))

# prediction (beta regression)
X_pred = sm.add_constant(adata.obs[src_train])
z_pred = 1 / (1 + np.exp(-X_pred.dot(reg.beta)))
adata.obs['latent_z_reg'] = z_pred
adata.write(os.path.join(pth_out, f'{name}.h5ad'))

#%%
