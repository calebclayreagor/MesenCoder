#%%
import os, glob
import pandas as pd
import pyranges as pr
from utils.ref import get_tss_pyranges

pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'unzip', 'GSE230319')
pth_ref = os.path.join(pth, 'ref')
pth_mod = os.path.join(pth, 'features', 'biomart', 'modules')
pth_out = os.path.join(pth, 'DELAY', 'late')

# load late module, TSS ranges
g = pd.read_csv(os.path.join(pth_mod, 'late.csv'))
g_dict = g.set_index('hsapiens').mmusculus.to_dict()
fn = os.path.join(pth_ref, 'hg38.refGene.gtf.gz')
tss = get_tss_pyranges(fn)

# compile edges
g1, g2 = [], []
for src in ('Twist1', 'Alx1'):
    pat = f'*{src.upper()}*'
    for fn in sorted(glob.glob(os.path.join(pth_in, pat))):
        print(src, fn)

        # assign peaks to TSSs
        peaks = pr.read_bed(fn)
        tss_peaks = peaks.nearest(tss, apply_strand_suffix = True)
        g_peaks = tss_peaks.df.gene_id.unique()
        tgt = [g_dict[_g_] for _g_ in g_peaks if _g_ in g_dict]
        g1.extend([src] * len(tgt)); g2.extend(tgt)

# save output
g_out = {'Gene1' : g1, 'Gene2' : g2}
refNet = (pd.DataFrame(g_out).drop_duplicates()
          .sort_values(['Gene1', 'Gene2']))
fn = os.path.join(pth_out, 'refNetwork.csv')
refNet.to_csv(fn, index = False)

# edge densities
n_edges = refNet.groupby('Gene1').size()
density = (n_edges / g.shape[0])
density.sort_values(ascending = False)

#%%
