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

pat_dict = {
    '*ALX1*'     : 'Alx1',
    'Brachyury*' : 'T',
    '*TWIST1*'   : 'Twist1'}

ref_dict = {
    'Alx1'   : 'hg38',
    'T'      : None,
    'Twist1' : 'hg38'}

# load genes, TSSs
g = pd.read_csv(os.path.join(pth_mod, 'union.csv'))
g_dict = g.set_index('hsapiens').mmusculus.to_dict()
fn = os.path.join(pth_ref, 'hg38.refGene.gtf.gz')
tss = get_tss_pyranges(fn)

# compile edges
g1, g2 = [], []
for pat, src in pat_dict.items():
    ref_src = ref_dict[src]
    for fn in sorted(glob.glob(os.path.join(pth_in, '*', pat))):
        print(src, fn)

        # assign peaks to TSS
        if ref_src is not None:
            peaks = pr.read_bed(fn)
            tss_peaks = peaks.nearest(tss, apply_strand_suffix = True)
            g_peaks = tss_peaks.df.gene_id.unique()
            tgt = [g_dict[_g_] for _g_ in g_peaks if _g_ in g_dict]
        else:
            g_peaks = pd.read_csv(fn, header = None)[0].unique()
            tgt = g_peaks[np.isin(g_peaks, g.mmusculus)].tolist()
        g1.extend([src] * len(tgt)); g2.extend(tgt)

# write output
fn = os.path.join(pth_out, 'refNetwork.csv')
refNet = (pd.DataFrame({'Gene1' : g1, 'Gene2' : g2})
          .drop_duplicates()
          .sort_values(['Gene1', 'Gene2'])
          .to_csv(fn, index = False))

#%%
# n_edges = refNet.groupby('Gene1').size()
# density = (n_edges / g.shape[0])
# density.sort_values(ascending = False)
