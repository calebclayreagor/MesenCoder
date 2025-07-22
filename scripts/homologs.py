#%%
import os, glob
import numpy as np
import pandas as pd
from pybiomart import Server
server = Server(host = 'http://www.ensembl.org')
mart = server.marts['ENSEMBL_MART_ENSEMBL']

#%%
mart_dict = {
    'mmusculus' : 'mmusculus_gene_ensembl',
    'hsapiens' : 'hsapiens_gene_ensembl'
    }

attr_dict = {
    'mmusculus' : {
        'external_gene_name'                    : 'mmusculus',
        'hsapiens_homolog_associated_gene_name' : 'hsapiens'
        },
    'hsapiens' : {
        'external_gene_name'                     : 'hsapiens',
        'mmusculus_homolog_associated_gene_name' : 'mmusculus'
        }
    }

for species in mart_dict:
    mart_dict[species] = mart.datasets[mart_dict[species]]
    mart_dict[species] = mart_dict[species].query(attributes = attr_dict[species].keys())
    mart_dict[species] = mart_dict[species].dropna().drop_duplicates()
    mart_dict[species].columns = attr_dict[species].values()
    # print(species, mart_dict[species])

#%%
summary_df = pd.read_csv(os.path.join('..', 'data', 'summary.csv'), index_col = 0)
summary_df = summary_df.loc[summary_df.Split == 'Training']
species_dict = summary_df.Species.to_dict()
feat_fn = sorted(glob.glob(os.path.join('..', 'data', 'features', '*.txt')))
feat_dict = {os.path.split(fn)[1].replace('.txt', '') : fn for fn in feat_fn}
outdir = os.path.join('..', 'data', 'features', 'biomart')
for i, key in enumerate(feat_dict):
    features = np.loadtxt(feat_dict[key], dtype = str)
    species = species_dict[key]
    df = mart_dict[species].copy()
    df = df.loc[df[species].isin(features)].reset_index(drop = True)
    df.to_csv(os.path.join(outdir, f'{key}.csv'), index = False)
    feat_dict[key] = df
    # print(key, df)

#%%
df = pd.concat(feat_dict.values()).sort_values('mmusculus')
df = df.apply(lambda x: x.drop_duplicates().reset_index(drop = True)).dropna()
df.to_csv(os.path.join(outdir, 'union.csv'), index = False)

#%%
