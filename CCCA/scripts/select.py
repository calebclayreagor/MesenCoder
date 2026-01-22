#%%
import os
import numpy as np
import pandas as pd
import scanpy as sc

names = ['Data_He2021_Prostate',
         'Data_Maynard2020_Lung',
         'Data_Puram2017_Head-and-Neck',
         'Data_Karaayvas2018_Breast',
         'Data_Filbin2018_Brain',
         'Data_Gojo2020_Brain',
         'Data_Hovestadt2019_Brain',
         'Data_Neftel2019_Brain',
         'Data_Tirosh2016_Brain',
         'Data_Venteicher2017_Brain',
         'Data_Jerby-Arnon2018_Skin']

pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'modeling', 'predict')
pth_mod = os.path.join(pth, 'features', 'biomart', 'modules')
pth_out = os.path.join(pth, 'modeling', 'landscape')

# load data
adata_dev = sc.read_h5ad(os.path.join(pth_out, 'development.h5ad'))
adata = sc.read_h5ad(os.path.join(pth_in, 'CCCA.h5ad'))

# early/late signatures
mod_names = ['Early', 'Late']
for mod in mod_names:
    print(mod)
    fn = os.path.join(pth_mod, f'{mod.lower()}.csv')
    g = pd.read_csv(fn).mmusculus.copy()
    g = g.loc[g.isin(adata.var_names)]
    for _, df_group in adata.obs.groupby('source', observed = True):
        msk = adata.obs_names.isin(df_group.index)
        adata_group = adata[msk].copy()
        sc.tl.score_genes(adata_group,
                          gene_list = g,
                          ctrl_as_ref = False,
                          score_name = mod,
                          random_state = 1234)
        adata.obs.loc[msk, mod] = adata_group.obs[mod]

# save basis
X = adata.obs[mod_names].values
adata.obsm['X_mod'] = X.copy()
msk_mod = (X.sum(axis = 1) > 0)

# early-late, latent axes => landscape
dims = ['X_mesen1', 'X_mesen2']
dX = (X[:, 1] - X[:, 0]).reshape(-1, 1)
z = adata.obs.latent_z.values.reshape(-1, 1)
X_mesen = np.concatenate((dX, z), axis = 1)
adata.obsm['X_mesen'] = X_mesen.copy()
adata.obs[dims] = X_mesen.copy()

# select developmental landscape
msk_dev = (adata_dev.obs.landscape == 'True')
X_ref = adata_dev[msk_dev].obsm['X_mesen'].copy()
scale = np.ptp(X_ref, axis = 0)
X_3ca = adata[msk_mod].obsm['X_mesen'].copy()
X_ref = X_ref[np.newaxis, :, :] / scale
X_3ca = X_3ca[:, np.newaxis, :] / scale
d = np.linalg.norm(X_ref - X_3ca, axis = 2)
msk_dist = (d <= .1).any(axis = 1)

# select sources
get_10x = lambda src: (os.path.split(src)[1] == '10X')
msk_name = adata[msk_mod].obs.Name.isin(names)
msk_10x = adata[msk_mod].obs.source.apply(get_10x)
msk_mesen = (msk_dist & msk_name & ~msk_10x)

# save 3CA landscape
landscape = np.full(adata.n_obs, 'False')
landscape[msk_mod] = msk_mesen.astype(str)
adata.obs['landscape'] = landscape
adata.write(os.path.join(pth_out, 'CCCA.h5ad'))

#%%
