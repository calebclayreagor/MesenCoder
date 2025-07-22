import os, glob
import pandas as pd
import anndata as ad
from scipy.io import mmread

def load_3CA_adata(data_dir: str) -> ad.AnnData:
    # get expression file (.mtx)
    mtx_fn_list = sorted(glob.glob(os.path.join(data_dir, '*.mtx')))
    if not mtx_fn_list:
        raise FileNotFoundError('No .mtx file found in the folder.')
    elif len(mtx_fn_list) > 1:
        raise ValueError('Multiple .mtx files found in the folder.')
    mtx_fn = mtx_fn_list[0]

    # get gene name file
    gene_fn_list = sorted(f for f in os.listdir(data_dir) if 'genes' in f.lower())
    if not gene_fn_list:
        raise FileNotFoundError("No file containing 'genes' found in the folder.")
    gene_fn = os.path.join(data_dir, gene_fn_list[0])

    # get cell name file (metadata)
    cell_fn_list = sorted(f for f in os.listdir(data_dir) if 'cells' in f.lower())
    if not cell_fn_list:
        raise FileNotFoundError("No file containing 'cells' found in the folder.")
    cell_fn = os.path.join(data_dir, cell_fn_list[0])

    # read files -> return adata
    X = mmread(mtx_fn).tocsr().T
    genes = pd.read_csv(gene_fn, header = None)
    genes.columns = ['genes']
    genes.set_index('genes', inplace = True)
    obs_df = pd.read_csv(cell_fn, header = 0)
    obs_df.set_index(obs_df.columns[0], inplace = True)
    return ad.AnnData(X = X, obs = obs_df, var = genes)
