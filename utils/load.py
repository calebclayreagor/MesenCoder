import os, glob
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread

def load_CCCA_adata(datadir: str, verbose: bool = True) -> ad.AnnData:
    if verbose:
        print('Loading', datadir)

    # get expression file (.mtx)
    mtx_list = sorted(glob.glob(os.path.join(datadir, '*.mtx')))
    if not mtx_list:
        raise FileNotFoundError('No .mtx file found in the folder.')
    elif len(mtx_list) > 1:
        raise ValueError('Multiple .mtx files found in the folder.')
    mtx_fn = mtx_list[0]

    # get gene file
    datadir_gene = datadir
    gene_list = sorted(f for f in os.listdir(datadir_gene) if 'genes' in f.lower() and '.mtx' not in f.lower())
    if not gene_list:
        datadir_gene = os.path.split(datadir)[0]
        gene_list = sorted(f for f in os.listdir(datadir_gene) if 'genes' in f.lower() and '.mtx' not in f.lower())
    if not gene_list:
        raise FileNotFoundError('No file containing genes found in the folders.')
    elif len(gene_list) > 1:
        raise ValueError('Multiple files containing genes found in the folders.')
    gene_fn = os.path.join(datadir_gene, gene_list[0])

    # get metadata file
    cell_list = sorted(f for f in os.listdir(datadir) if 'cells' in f.lower() and '.mtx' not in f.lower())
    if not cell_list:
        raise FileNotFoundError('No file containing cells found in the folder.')
    elif len(cell_list) > 1:
        raise ValueError('Multiple files containing cells found in the folder.')
    cell_fn = os.path.join(datadir, cell_list[0])

    # read files
    X = mmread(mtx_fn).tocsr().T
    genes = pd.read_csv(gene_fn, header = None)
    genes.columns = ['genes']
    genes.set_index('genes', inplace = True)
    obs = pd.read_csv(cell_fn, header = 0, dtype = str)
    obs.set_index(obs.columns[0], inplace = True)
    adata = ad.AnnData(X = X, obs = obs, var = genes)
    adata.var_names_make_unique()

    # normalize/scale expression
    if 'tpm' in mtx_fn.lower():
        pass
    elif 'tp10k' in mtx_fn.lower():
        pass
    elif 'normalized' in mtx_fn.lower():
        pass
    elif 'counts' in mtx_fn.lower():
        sc.pp.normalize_total(adata)
    else:
        raise ValueError('Expression values in .mtx are not normalized or counts.')
    sc.pp.log1p(adata)

    if verbose:
        print('Done.')

    return adata
