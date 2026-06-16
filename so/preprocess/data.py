#!/usr/bin/env python
import numpy as np
import pandas as pd
import copy
import torch
import scanpy as sc
import scipy.sparse as sp
# import cv2
import scipy
import sklearn
from typing import Any
from scipy.sparse import csr, csr_matrix
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from anndata import AnnData
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors


def minmax_scale(adata):
    data_min = adata.X.min(axis=1).toarray().flatten()
    data_max = adata.X.max(axis=1).toarray().flatten()
    scale_factor = data_max - data_min
    scale_factor[scale_factor == 0] = 1
    row_indices = adata.X.nonzero()[0]
    adata.X.data = 10 * (adata.X.data - data_min[row_indices]) / scale_factor[row_indices]
    return adata


def process_adata(data_list,
                  label='batch',
                  keys=None,
                  join='inner',
                  n_top_features=None,
                  spatial_key='spatial',
                  coordinate_dimension=2,
                  filter_hk_genes=False,
                  norm=True,
                  target_sum=1e4,
                  scale='log',
                  ):
    """
    :param data_list:
    :param label:
    :param keys:
    :param join:
    :param n_top_features:
    :param spatial_key:
    :param coordinate_dimension:
    :param filter_hk_genes:
    :param norm:
    :param target_sum:
    :param scale:
    :return:
    """
    if keys is None:
        keys = list(range(len(data_list)))
        keys = ['batch' + key for key in keys]
    for i, temp in enumerate(data_list):
        temp.obs_names_make_unique()
        temp.var_names_make_unique()
        # reset index
        temp.obs = temp.obs.reset_index(drop=True)
        temp.obs.loc[:, 'original_index'] = temp.obs.index
        temp.obs.index = [str(keys[i]) + '_' + str(j) for j in temp.obs.index]
        if not isinstance(temp.X, csr.csr_matrix):
            data_list[i].X = csr_matrix(temp.X)
        if filter_hk_genes:
            temp = temp[:, [gene for gene in temp.var_names if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
        if 'spot_quality' not in temp.obs.columns:
            temp.obs.loc[:, 'spot_quality'] = 'real'
        if spatial_key in temp.obsm.keys():
            pass
            # coordinates = temp.obsm['spatial']
            # max_value = np.max(coordinates)
            # normalized_coordinates = coordinates / max_value
            # temp.obsm['spatial'] = normalized_coordinates + i
        else:
            num_points = len(temp)
            x = np.random.uniform(low=0.5, high=0.6, size=num_points)
            y = np.random.uniform(low=0.5, high=0.6, size=num_points)
            pseudo_coordinates = np.vstack((x, y)).T
            if coordinate_dimension == 3:  # 3d
                z = np.random.uniform(low=0.5, high=0.6, size=num_points)
                pseudo_coordinates = np.vstack((x, y, z)).T
            temp.obsm['spatial'] = pseudo_coordinates
            print(f'warning! spatial is not in {i}th adata.obsm')

    adata = sc.concat([*data_list], label=label, keys=keys, join=join)
    if len(adata.var) == 0:
        raise ValueError('No concat gene')
    adata.obs[label] = adata.obs[label].astype('category')

    # counts
    adata.layers['counts'] = copy.deepcopy(adata.X)
    # choose real to select hvg
    if norm:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if scale == 'log':
        sc.pp.log1p(adata)
    elif scale == 'z':
        sc.pp.scale(adata, zero_center=False, max_value=10)
    elif scale == 'minmax':
        temp = sc.concat([minmax_scale(data.copy()) for data in data_list])
        adata.X = temp.X
    if norm and scale:
        adata.layers['norm_log'] = copy.deepcopy(adata.X)
    adata.raw = adata
    # hvg
    if n_top_features:
        temp_real = adata[adata.obs.loc[:, 'spot_quality'] == 'real', :]
        sc.pp.highly_variable_genes(temp_real, n_top_genes=n_top_features, batch_key=label, flavor='seurat_v3', layer='counts')
        adata.var = temp_real.var
        adata = adata[:, adata.var.highly_variable]
    # save mean and var
    B, G = len(keys), len(adata.var)
    mu_bg = np.zeros((B, G), dtype=np.float32)
    std_bg = np.zeros((B, G), dtype=np.float32)
    for b, key in enumerate(keys):
        temp = adata[adata.obs['batch'] == key, :]
        X = temp.X.toarray() if isinstance(temp.X, csr.csr_matrix) else np.asarray(temp.X)
        mu_bg[b] = X.mean(axis=0)
        std_bg[b] = X.std(axis=0)
        std_bg[b][std_bg[b] < 1e-6] = 1e-6

    if not isinstance(adata.X, csr.csr_matrix):
        adata.X = csr_matrix(adata.X)

    adata.uns['mu_bg'] = mu_bg
    adata.uns['std_bg'] = std_bg
    return adata


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
        adata=None,
        n_components=50,
        use_highly_variable=None,
        random_state=42,
):
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)  # TODO difference in win10 and linux ubuntu 22.04
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, random_state=random_state)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def cal_spatial_net(adata: AnnData,
                    cutoff: [int, float] = None,
                    max_neigh: int = 100,
                    metric='euclidean',
                    model: str = 'Radius',
                    spatial_key: str = 'spatial',
                    verbose: bool = True) -> tuple[AnnData, Any]:
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata : AnnData
        AnnData object of scanpy package.
    cutoff : float, optional
        Radius cutoff when model='Radius'.
    max_neigh: int
        The max neighbors of KNN
    metric: str
        euclidean or cosine
    model : str
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less
        than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    spatial_key: str
        The key of spatial coordinates in adata.obsm
    verbose : bool
        Whether to print progress messages.

    """
    assert model in ['Radius', 'KNN'], "Model must be either 'Radius' or 'KNN'"
    from sklearn.neighbors import NearestNeighbors

    if verbose:
        print('------Calculating spatial graph...')

    coor = pd.DataFrame(adata.obsm[spatial_key], index=adata.obs.index)
    nbrs = NearestNeighbors(n_neighbors=max_neigh + 1, algorithm='ball_tree', metric=metric).fit(coor)
    distances, indices = nbrs.kneighbors(coor)

    if model == 'KNN':
        indices = indices[:, 1:cutoff + 1]
        distances = distances[:, 1:cutoff + 1]
    else:  # model == 'Radius'
        mask = distances[:, 1:] < cutoff
        indices = indices[:, 1:]
        distances = distances[:, 1:]
        indices[~mask] = -1
        distances[~mask] = -1

    valid_mask = indices.flatten() != -1
    KNN_df = pd.DataFrame({
        'Cell1': np.repeat(np.arange(coor.shape[0]), indices.shape[1])[valid_mask],
        'Cell2': indices.flatten()[valid_mask],
        'Distance': distances.flatten()[valid_mask]
    })

    id_cell_trans = dict(enumerate(coor.index))
    KNN_df['Cell1'] = KNN_df['Cell1'].map(id_cell_trans)
    KNN_df['Cell2'] = KNN_df['Cell2'].map(id_cell_trans)

    if model == 'Radius':
        Spatial_Net = KNN_df[KNN_df['Distance'] < cutoff]
    else:
        Spatial_Net = KNN_df

    if verbose:
        print(f'The graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{Spatial_Net.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')

    # adata.uns['Spatial_Net'] = Spatial_Net

    # Create adjacency matrix
    cell_indices = pd.Series(range(adata.n_obs), index=adata.obs.index)
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cell_indices)
    G_df['Cell2'] = G_df['Cell2'].map(cell_indices)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # Add self-loops

    adata.uns['adj'] = G

    return adata, G


def process_graph(adata, data_list, cutoff_list=None, max_neigh=100, spatial_key='spatial', metric='euclidean',
                  model='Radius', verbose=True):
    if adata.obsm[spatial_key].shape[1] == 2:  # 2d, construct graph in each data_list
        from scipy.sparse import block_diag
        print('cal spatial net in data_list')
        data_list = [
            cal_spatial_net(adata_temp, cutoff=cutoff_list[i], max_neigh=max_neigh, spatial_key=spatial_key,
                            metric=metric, model=model, verbose=verbose)[0]
            for i, adata_temp in enumerate(data_list)]
        adj_list = [item.uns['adj'] for item in data_list]
        adj_concat = block_diag(adj_list)
        adata.uns['adj'] = adj_concat
    elif adata.obsm[spatial_key].shape[1] >= 3:  # 3d, construct graph in one adata
        print('cal spatial net in one adata')
        adata, adj_concat = cal_spatial_net(adata, cutoff=cutoff_list[0], max_neigh=max_neigh, spatial_key=spatial_key,
                                            metric=metric, model=model, verbose=verbose)
    else:
        raise NotImplementedError
    return adata, adj_concat


def cell_type_onehot_encoding(adata, cell_type_key='cell_type'):
    categories = adata.obs.loc[:, cell_type_key].to_numpy()
    encoder = OneHotEncoder()
    categories_reshaped = categories.reshape(-1, 1)
    one_hot_matrix = encoder.fit_transform(categories_reshaped).todense()
    one_hot_columns = encoder.categories_[0]
    adata_new = sc.AnnData(one_hot_matrix, obs=adata.obs, obsm=adata.obsm)
    adata_new.var.index = one_hot_columns
    return adata_new


def compute_global_avg_knn_distance(adata, spatial_key='spatial', k=2):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(adata.obsm[spatial_key])  # k+1因为包含自己
    distances, indices = nbrs.kneighbors(adata.obsm[spatial_key])
    individual_avgs = np.mean(distances[:, 1:], axis=1)  # 从第二列开始，因为第一列是自己
    global_avg = np.mean(individual_avgs)

    return global_avg, individual_avgs
