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
    'Cdx2*'      : 'Cdx2',
    'Sox2*'      : 'Sox2',
    'Brachyury*' : 'T',
    '*TWIST1*'   : 'Twist1'}

ref_dict = {
    'Alx1'   : 'hg38',
    'Cdx2'   : 'mm9',
    'Sox2'   : None,
    'T'      : None,
    'Twist1' : 'hg38'}

# load genes
g = pd.read_csv(os.path.join(pth_mod, 'union.csv'))
g_dict = g.set_index('hsapiens').mmusculus.to_dict()

# TSS reference ranges
fn_ref = sorted(glob.glob(os.path.join(pth_ref, '*')))
get_ref = lambda fn: os.path.split(fn)[1].split('.')[0]
tss_dict = {get_ref(fn) : get_tss_pyranges(fn) for fn in fn_ref}

# compile edges
g1, g2 = [], []
for pat, src in pat_dict.items():
    for fn in sorted(glob.glob(os.path.join(pth_in, '*', pat))):
        ref_src = ref_dict[src]
        print(src, ref_src, fn)

        # assign peaks to TSS
        if ref_src is not None:
            peaks = pr.read_bed(fn)
            tss_src = tss_dict[ref_src]
            tss_peaks = peaks.nearest(tss_src, apply_strand_suffix = True)
            g_peaks = tss_peaks.df.gene_id.unique()
        else:
            g_peaks = pd.read_csv(fn, header = None)[0].unique()

        # select targets
        if ref_src == 'hg38':
            tgt = [g_dict[_g_] for _g_ in g_peaks if _g_ in g_dict]
        else:
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
