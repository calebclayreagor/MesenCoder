import pandas as pd
import pyranges as pr
from typing import Optional, Dict

def get_tss_pyranges(pth_ref: str) -> pr.PyRanges:
    ref = pr.read_gtf(pth_ref)
    tss = (ref[ref.Feature == 'transcript'].df
           .groupby('gene_id', group_keys = False)
           .apply(get_upstream_tss))
    df = pd.DataFrame(tss[~tss.isna()].tolist())
    return pr.PyRanges(df)

def get_upstream_tss(sdf: pd.DataFrame) -> Optional[Dict]:
    chrom = sdf.Chromosome.unique()
    strand = sdf.Strand.unique()
    if (len(chrom) == 1) and (len(strand) == 1):
        gene = sdf.gene_id.iloc[0]
        if strand[0] == '+':
            tss = sdf.Start.min()
        else:
            tss = sdf.End.max() - 1
        return {'Chromosome': chrom[0],
                'Start': tss,
                'End': tss + 1,
                'Strand': strand[0],
                'gene_id': gene}