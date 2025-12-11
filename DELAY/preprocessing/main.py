#%%
import os
import pandas as pd
import scanpy as sc

pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'modeling', 'predict')
pth_feat = os.path.join(pth, 'features')
pth_mod = os.path.join(pth_feat, 'biomart', 'modules')
pth_tf = os.path.join(pth_feat, 'AnimalTFDB4')
pth_out = os.path.join(pth, 'DELAY')

g = pd.read_csv(os.path.join(pth_mod, 'union.csv')).mmusculus
fn = os.path.join(pth_tf, 'Mus_musculus_TF.txt')
tf_ref = pd.read_csv(fn, sep = '\t').Symbol

# select trajectories
src = ('GSE162534', 'GSE229103', 'rRNAModifications')
adata = sc.read_h5ad(os.path.join(pth_in, 'development.h5ad'))
msk_traj = (adata.obs.trajectory == 'True')
msk_src = adata.obs.source.isin(src)
adata = adata[(msk_traj & msk_src), g]

# save outputs
X = pd.DataFrame(adata.X.toarray(), index = adata.obs_names,
                 columns = adata.var_names)
z = adata.obs.latent_z.rename('PseudoTime').astype(float)
tf = adata.var_names.to_series().loc[adata.var_names.isin(tf_ref)]
X.T.to_csv(os.path.join(pth_out, 'NormalizedData.csv'))
z.to_csv(os.path.join(pth_out, 'PseudoTime.csv'))
fn = os.path.join(pth_out, 'TranscriptionFactors.csv')
tf.to_csv(fn, index = False, header = False)

#%%
