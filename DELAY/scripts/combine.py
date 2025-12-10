#%%
import os, glob
import pandas as pd

top_n = 20
pth = os.path.join('..', '..', 'data', 'DELAY')
pth_out = os.path.join(pth, 'combined')
pth_reg = os.path.join(pth_out, 'regulons')

# load predictions
pred_fn = sorted(glob.glob(os.path.join(pth, '*', 'regPredictions.csv')))
get_geo = lambda fn: os.path.split(os.path.split(fn)[0])[1]
get_pred = lambda fn: pd.read_csv(fn, index_col = 0)
pred_dict = {get_geo(fn) : get_pred(fn) for fn in pred_fn}

# combine predictions
pred_early = (pred_dict['GSE162534'] + pred_dict['GSE229103']) / 2
pred_avg = (pred_early + pred_dict['rRNAModifications']) / 2
pred_avg.index = pred_avg.index.str.capitalize()
pred_avg.columns = pred_avg.columns.str.capitalize()
pred_avg.to_csv(os.path.join(pth_out, 'regPredictions.csv'))

# top N network
pred_rank = pred_avg.rank(axis = 0, ascending = False)
pred_grn = ((pred_rank <= top_n) & (pred_avg > .5)).astype(int)
pred_grn.to_csv(os.path.join(pth_out, f'networkTop{top_n}.csv'))

# top N hubs
hubs = (pred_grn.sum(axis = 1)
        .sort_values(ascending = False)
        .rename('Outdegree'))
fn = f'hubsTop{top_n}.csv'
hubs.to_csv(os.path.join(pth_out, fn))

# regulons (top hubs)
for g in ('Twist1', 'Prrx2'):
    msk_tgt = pred_grn.loc[g].astype(bool)
    tgt = pred_grn.columns[msk_tgt].to_series()
    fn = os.path.join(pth_reg, f'{g}.csv')
    tgt.to_csv(fn, header = False, index = False)

#%%
