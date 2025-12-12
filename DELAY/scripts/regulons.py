#%%
import os
import pandas as pd

pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'DELAY')
pth_feat = os.path.join(pth, 'features', 'biomart')
pth_out = os.path.join(pth_in, 'regulons')

g = pd.read_csv(os.path.join(pth_feat, 'union.csv'))

# load GRN
fn = os.path.join(pth_in, 'networkTop20.csv')
A = pd.read_csv(fn, index_col = 0).astype(bool)
A.index = A.index.str.capitalize()
A.columns = A.columns.str.capitalize()
A = A.rename(columns = {'1700017b05rik' : '1700017B05Rik'})

# save regulons
for ix in A.index:
    tgt = A.columns[A.loc[ix]]
    if tgt.size > 0:
        reg = g.loc[g.mmusculus.isin(tgt)]
        fn = os.path.join(pth_out, f'{ix}.csv')
        reg.to_csv(fn, index = False)

#%%