#%%
import os
import pandas as pd

pth = os.path.join('..', '..', 'data')
pth_in = os.path.join(pth, 'DELAY')
pth_feat = os.path.join(pth, 'features', 'biomart')
pth_out = os.path.join(pth_in, 'regulons')

g = pd.read_csv(os.path.join(pth_feat, 'union.csv'))
reg = ('Gata4', 'Twist1', 'Snai1', 'Foxc1', 'Prrx2')

# load GRN
fn = os.path.join(pth_in, 'networkTop20.csv')
A = pd.read_csv(fn, index_col = 0).astype(bool)
A.index = A.index.str.capitalize()
A.columns = A.columns.str.capitalize()
A = A.rename(columns = {'1700017b05rik' : '1700017B05Rik'})

# save regulons
for name in reg:
    tgt = A.columns[A.loc[name]]
    g_out = g.loc[g.mmusculus.isin(tgt)]
    fn = os.path.join(pth_out, f'{name}.csv')
    g_out.to_csv(fn, index = False)

#%%