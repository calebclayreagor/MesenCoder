#%%
import os
import pandas as pd
import scanpy as sc

# params
name = 'CCCA'

# load data
pth = os.path.join('..', '..', 'data')
pth_pred = os.path.join(pth, 'modeling', 'predict')
pth_out = os.path.join(pth, 'modeling', 'attribution')
adata = sc.read_h5ad(os.path.join(pth_pred, f'{name}.h5ad'))
df = pd.read_csv(os.path.join(pth, 'summary.csv'), index_col = 0)

# attribution (X * beta)
src_train = df.loc[df.Training == True].index
params = pd.read_csv('params.csv', index_col = 0)
X = adata.obs[src_train].copy()
beta = params.loc[src_train].values.T
attr = X.multiply(beta, axis = 1)
adata.obsm['attributions'] = attr

# save
adata.write(os.path.join(pth_out, f'{name}.h5ad'))

#%%
