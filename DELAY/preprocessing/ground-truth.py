#%%
import os, glob
import numpy as np
import pandas as pd
import pyranges as pr
from utils.ref import get_tss_pyranges

pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'unzip')
pth_ref = os.path.join(pth, 'ref')
pth_mod = os.path.join(pth, 'features', 'biomart', 'modules')
pth_out = os.path.join(pth, 'DELAY')
pth_early = os.path.join(pth_out, 'early')
pth_late = os.path.join(pth_out, 'late')

src_early = ['T', 'Eomes']
src_late = ['Alx1', 'Twist1']
pat_dict = {
    '*ALX1*'     : 'Alx1',
    '*TWIST1*'   : 'Twist1',
    'Brachyury*' : 'T',
    'Eomes*'     : 'Eomes'}

# load modules, TSSs
early = pd.read_csv(os.path.join(pth_mod, 'early.csv'))
late = pd.read_csv(os.path.join(pth_mod, 'late.csv'))
g_dict = late.set_index('hsapiens').mmusculus.to_dict()
fn = os.path.join(pth_ref, 'hg38.refGene.gtf.gz')
tss = get_tss_pyranges(fn)

# compile edges
g1_early, g2_early = [], []
g1_late, g2_late = [], []
for pat, src in pat_dict.items():
    for fn in sorted(glob.glob(os.path.join(pth_in, '*', pat))):
        print(src, fn)

        # early module
        if src in src_early:
            g_peaks = pd.read_csv(fn, header = None)[0].unique()
            tgt = g_peaks[np.isin(g_peaks, early.mmusculus)].tolist()
            g1_early.extend([src] * len(tgt)); g2_early.extend(tgt)

        # late module
        elif src in src_late:
            # assign peaks to TSS
            peaks = pr.read_bed(fn)
            tss_peaks = peaks.nearest(tss, apply_strand_suffix = True)
            g_peaks = tss_peaks.df.gene_id.unique()
            tgt = [g_dict[_g_] for _g_ in g_peaks if _g_ in g_dict]
            g1_late.extend([src] * len(tgt)); g2_late.extend(tgt)       

# ground truth (early)
refNet_early = (pd.DataFrame({'Gene1' : g1_early, 'Gene2' : g2_early})
                .drop_duplicates().sort_values(['Gene1', 'Gene2']))
refNet_early.to_csv(os.path.join(pth_early, 'refNetwork.csv'), index = False)

# ground truth (late)
refNet_late = (pd.DataFrame({'Gene1' : g1_late, 'Gene2' : g2_late})
               .drop_duplicates().sort_values(['Gene1', 'Gene2']))
refNet_late.to_csv(os.path.join(pth_late, 'refNetwork.csv'), index = False)

#%%
# n_edges = refNet_late.groupby('Gene1').size()
# density = (n_edges / late.shape[0])
# density.sort_values(ascending = False)
