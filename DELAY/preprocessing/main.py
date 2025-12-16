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
pth_early = os.path.join(pth_out, 'early')
pth_late = os.path.join(pth_out, 'late')

early = pd.read_csv(os.path.join(pth_mod, 'early.csv'))
late = pd.read_csv(os.path.join(pth_mod, 'late.csv'))
fn = os.path.join(pth_tf, 'Mus_musculus_TF.txt')
tf_ref = pd.read_csv(fn, sep = '\t').Symbol

# select trajectories (pooling)
src = ('GSE162534', 'GSE229103', 'rRNAModifications')
adata = sc.read_h5ad(os.path.join(pth_in, 'development.h5ad'))
msk_traj = (adata.obs.trajectory == 'True')
msk_src = adata.obs.source.isin(src)
adata = adata[(msk_traj & msk_src)]

# balance data (random sampling)
grp = adata.obs.groupby('source', observed = True)
rand = grp.sample(n = grp.size().min(), random_state = 1)
adata = adata[adata.obs_names.isin(rand.index)]

# select data (all)
X = pd.DataFrame(adata.X.toarray(),
                 index = adata.obs_names,
                 columns = adata.var_names)
z = adata.obs.latent_z.rename('PseudoTime')
tf = (adata.var_names.to_series()
      .loc[adata.var_names.isin(tf_ref)])

# early module
X_early = X.loc[:, X.columns.isin(early.mmusculus)].T
tf_early = tf.loc[tf.isin(early.mmusculus)]
X_early.to_csv(os.path.join(pth_early, 'NormalizedData.csv'))
z.to_csv(os.path.join(pth_early, 'PseudoTime.csv'))
fn = os.path.join(pth_early, 'TranscriptionFactors.csv')
tf_early.to_csv(fn, index = False, header = False)

# late module
X_late = X.loc[:, X.columns.isin(late.mmusculus)].T
tf_late = tf.loc[tf.isin(late.mmusculus)]
X_late.to_csv(os.path.join(pth_late, 'NormalizedData.csv'))
z.to_csv(os.path.join(pth_late, 'PseudoTime.csv'))
fn = os.path.join(pth_late, 'TranscriptionFactors.csv')
tf_late.to_csv(fn, index = False, header = False)

#%%
