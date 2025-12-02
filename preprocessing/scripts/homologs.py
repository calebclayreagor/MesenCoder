#%%
import os, glob
import numpy as np
import pandas as pd
from pybiomart import Server
server = Server(host = 'http://www.ensembl.org')
mart = server.marts['ENSEMBL_MART_ENSEMBL']

pth = os.path.join('..', '..')
pth_data = os.path.join(pth, 'data')
outdir = os.path.join(pth_data, 'features', 'biomart')

# datasets summary
summary_df = pd.read_csv(os.path.join(pth_data, 'summary.csv'), index_col = 0)
species_dict = summary_df.Species.to_dict()

# dataset features
feat_fn = sorted(glob.glob(os.path.join(pth_data, 'features', '*.txt')))
feat_dict = {os.path.split(fn)[1].replace('.txt', '') : fn for fn in feat_fn}

mart_dict = {
    'mmusculus' : 'mmusculus_gene_ensembl',
    'hsapiens' : 'hsapiens_gene_ensembl'}

attr_dict = {
    'mmusculus' : {
        'external_gene_name'                     : 'mmusculus',
        'hsapiens_homolog_associated_gene_name'  : 'hsapiens'},
    'hsapiens' : {
        'external_gene_name'                     : 'hsapiens',
        'mmusculus_homolog_associated_gene_name' : 'mmusculus'}}

for species in mart_dict:
    mart_dict[species] = mart.datasets[mart_dict[species]]
    mart_dict[species] = mart_dict[species].query(attributes = attr_dict[species].keys())
    mart_dict[species] = mart_dict[species].dropna().drop_duplicates()
    mart_dict[species].columns = attr_dict[species].values()
    # print(species, mart_dict[species])

for i, key in enumerate(feat_dict):
    features = np.loadtxt(feat_dict[key], dtype = str)
    species = species_dict[key]
    df = mart_dict[species].copy()
    df = df.loc[df[species].isin(features)].reset_index(drop = True)
    df = df.sort_values(['mmusculus', 'hsapiens']).reset_index(drop = True)
    df = df.drop_duplicates(subset = ['mmusculus'], keep = 'first')
    df = df.drop_duplicates(subset = ['hsapiens'], keep = 'first')
    df = df.sort_values('mmusculus').reset_index(drop = True)
    df.to_csv(os.path.join(outdir, f'{key}.csv'), index = False)
    feat_dict[key] = df
    # print(key, df)

df = pd.concat(feat_dict.values(), ignore_index = True).dropna()
df = df.sort_values(['mmusculus', 'hsapiens']).reset_index(drop = True)
df = df.drop_duplicates(subset = ['mmusculus'], keep = 'first')
df = df.drop_duplicates(subset = ['hsapiens'], keep = 'first')
df = df.sort_values('mmusculus').reset_index(drop = True)
df.to_csv(os.path.join(outdir, 'union.csv'), index = False)

#%%
