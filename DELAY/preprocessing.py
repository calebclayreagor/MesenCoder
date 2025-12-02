#%%
import os
import pandas as pd
import scanpy as sc

pth = os.path.join('..', 'data')
pth_in = os.path.join(pth, 'modeling', 'inputs')
pth_feat = os.path.join(pth, 'features')
pth_mod = os.path.join(pth_feat, 'biomart', 'modules')
pth_tf = os.path.join(pth_feat, 'AnimalTFDB4')
pth_out = os.path.join(pth, 'DELAY')
names = ('GSE162534', 'GSE229103', 'rRNAModifications')

# load data
adata = sc.read_h5ad(os.path.join(pth_in, 'development.h5ad'))
g = pd.read_csv(os.path.join(pth_mod, 'union.csv')).mmusculus
tf_ref = pd.read_csv(os.path.join(pth_tf, 'Mus_musculus_TF.txt'), sep = '\t')

# write outputs
msk_traj = (adata.obs.trajectory == 'True')
tf = g.loc[g.isin(tf_ref.Symbol)]
for src in names:
    pth_src = os.path.join(pth_out, src)
    msk_src = (adata.obs.source == src)
    adata_src = adata[(msk_traj & msk_src), g].copy()

    # normalized expression
    X = pd.DataFrame(adata_src.X.toarray(),
                     index = adata_src.obs_names,
                     columns = adata_src.var_names)
    X.T.to_csv(os.path.join(pth_src, 'NormalizedData.csv'))

    # pseudotime
    t = adata_src.obs.t.rename('PseudoTime').astype(float)
    t.to_csv(os.path.join(pth_src, 'PseudoTime.csv'))

    # TFs
    fn = os.path.join(pth_src, 'TranscriptionFactors.csv')
    tf.to_csv(fn, index = False, header = False)

#%%
