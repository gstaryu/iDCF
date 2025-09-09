# -*- coding: UTF-8 -*-
"""
@Project: BIND
@File   : utils.py
@IDE    : PyCharm
@Author : hjguo
@Date   : 2025/7/9 11:37
@Doc    : Data processing and knowledge base loading code
"""
import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from typing import Union, List
import warnings

warnings.filterwarnings("ignore")


def load_data(data_name=None):
    """
    Load training and testing data based on the specified dataset name.

    :param data_name: Name of the dataset to be loaded.
    :return: Training and testing data in AnnData format.
    """
    if data_name == 'brain_human' or data_name == 'brain_human_541':
        data = sc.read_h5ad('./data/simu_data/PFC_norm_count_my_500cell_simu_8k.h5ad')
    elif data_name == 'HNSC':
        data = sc.read_h5ad('./data/simu_data/HNSC_my_500cell_simu.h5ad')
    elif data_name == 'HGSC':
        data = sc.read_h5ad('./data/simu_data/HGSC_GSE232314_my_500cell_simu.h5ad')
    elif data_name == 'CRC-sEV' or data_name == 'GSE100063-CRC-sEV' or data_name == 'GTEx_colon_sigmoid':
        data = sc.read_h5ad('./data/simu_data/CRC-sEV_my_500cell_simu.h5ad')
    elif data_name == 'Islet':
        data = sc.read_h5ad('./data/simu_data/Islet_my_500cell_simu.h5ad')
    elif data_name == 'Pancreas':
        data = sc.read_h5ad('./data/simu_data/Pancreas_my_500cell_simu.h5ad')
    else:
        data = sc.read_h5ad('./data/simu_data/data8k_norm_count_my_500cell_simu.h5ad')
        # data = sc.read_h5ad('./data/simu_data/new_data8k_norm_count_my_500cell_simu.h5ad')
        data.obs['Unknown'] = data.obs['Unknown'] + data.obs['Dendritic']
        data.obs.drop('Dendritic', axis=1, inplace=True)
        sorted_index = data.var_names.sort_values()
        data = data[:, sorted_index]
    data.uns = {'cell_types': list(data.obs.columns)}
    train_data = data

    idx = data.obs_names.tolist()
    col = data.var_names.tolist()
    train_x = pd.DataFrame(data.X, index=idx, columns=col)

    test_name = data_name
    if test_name == 'microarray':
        test_x = pd.read_csv('./data/Real bulk/' + test_name + '.txt', sep='\t', index_col=0)
        test_x = 2 ** test_x - 1
        test_x[test_x < 0] = 0
    elif test_name == 'brain_human':
        test_x = pd.read_csv('./data/Real bulk/ROSMAP_human_GEP.csv', index_col=0).T
    else:
        test_x = pd.read_csv('./data/Real bulk/' + test_name + '.txt', sep='\t', index_col=0).T

    print(f'Original train_x shape: {train_x.shape}, test_x shape: {test_x.shape}')

    print('Cutting variance...')
    # Remove all-zero genes
    train_x = train_x.loc[:, (train_x != 0).any(axis=0)]
    test_x = test_x.loc[:, (test_x != 0).any(axis=0)]
    # variance_threshold = 0.99999
    variance_threshold = 0.99
    var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[
        int(train_x.shape[
                1] * variance_threshold)]  # Calculate variance for each gene, sort from high to low, find genes ranked in the top variance_threshold
    train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]  # Remove genes with variance less than var_cutoff
    var_cutoff = test_x.var(axis=0).sort_values(ascending=False)[int(test_x.shape[1] * variance_threshold)]
    test_x = test_x.loc[:, test_x.var(axis=0) > var_cutoff]

    print(f'Filtered train_x genes: {train_x.shape[1]}, test_x genes: {test_x.shape[1]}')

    inter = train_x.columns.intersection(test_x.columns)
    train_x = train_x[inter]
    test_x = test_x[inter]

    print(f'Intersected genes: {len(inter)}')

    train_x = np.log2(train_x + 1)
    test_x = np.log2(test_x + 1)

    mms = MinMaxScaler()
    mms_train_x = mms.fit_transform(train_x.T).T
    mms_test_x = mms.fit_transform(test_x.T).T

    train_x = pd.DataFrame(mms_train_x, index=train_x.index, columns=train_x.columns)
    test_x = pd.DataFrame(mms_test_x, index=test_x.index, columns=test_x.columns)

    test_data = ad.AnnData(X=test_x.values)
    test_data.var_names = test_x.columns
    test_data.obs_names = test_x.index
    test_data.uns = {'cell_types': list(data.obs.columns)}

    train_data = ad.AnnData(X=train_x.values, obs=train_data.obs)
    train_data.var_names = train_x.columns
    train_data.uns = data.uns

    return train_data, test_data


def load_data_from_path(train_path, test_path):
    """
    Load training and testing data from specified file paths.

    :param train_path: Path to the training data file (.h5ad format).
    :param test_path: Path to the testing data file (.h5ad, .csv, or .txt format).
    :return: Training and testing data in AnnData format.
    """
    if train_path.endswith('.h5ad'):
        data = sc.read_h5ad(train_path)
    else:
        raise ValueError('Unsupported train data format. Please provide .h5ad file.')
    data.uns = {'cell_types': list(data.obs.columns)}
    train_data = data

    idx = data.obs_names.tolist()
    col = data.var_names.tolist()
    train_x = pd.DataFrame(data.X, index=idx, columns=col)

    if test_path.endswith('.h5ad'):
        test_data = sc.read_h5ad(test_path)
        idx = test_data.obs_names.tolist()
        col = test_data.var_names.tolist()
        test_x = pd.DataFrame(test_data.X, index=idx, columns=col)
    elif test_path.endswith('.csv'):
        test_x = pd.read_csv(test_path, index_col=0).T
    elif test_path.endswith('.txt'):
        test_x = pd.read_csv(test_path, sep='\t', index_col=0).T
    else:
        raise ValueError('Unsupported test data format. Please provide .h5ad, .csv, or .txt file.')

    print('Cutting variance...')
    # Remove all-zero genes
    train_x = train_x.loc[:, (train_x != 0).any(axis=0)]
    test_x = test_x.loc[:, (test_x != 0).any(axis=0)]
    # variance_threshold = 0.99999
    variance_threshold = 0.99
    var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[
        int(train_x.shape[
                1] * variance_threshold)]  # Calculate variance for each gene, sort from high to low, find genes ranked in the top variance_threshold
    train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]  # Remove genes with variance less than var_cutoff
    var_cutoff = test_x.var(axis=0).sort_values(ascending=False)[int(test_x.shape[1] * variance_threshold)]
    test_x = test_x.loc[:, test_x.var(axis=0) > var_cutoff]

    inter = train_x.columns.intersection(test_x.columns)
    train_x = train_x[inter]
    test_x = test_x[inter]

    train_x = np.log2(train_x + 1)
    test_x = np.log2(test_x + 1)

    mms = MinMaxScaler()
    mms_train_x = mms.fit_transform(train_x.T).T
    mms_test_x = mms.fit_transform(test_x.T).T

    train_x = pd.DataFrame(mms_train_x, index=train_x.index, columns=train_x.columns)
    test_x = pd.DataFrame(mms_test_x, index=test_x.index, columns=test_x.columns)

    test_data = ad.AnnData(X=test_x.values)
    test_data.var_names = test_x.columns

    train_data = ad.AnnData(X=train_x.values, obs=train_data.obs)
    train_data.var_names = train_x.columns
    train_data.uns = data.uns

    return train_data, test_data


def get_ppi(ppi_file_path, gene_list, one_hot=True):
    """
    Extract the protein-protein interaction (PPI) network for a specified list of genes from a PPI data file.

    :param ppi_file_path: PPI data file path
    :param gene_list: List of gene names
    :param one_hot: If True, use one-hot encoding for interactions; if False, use combined_score/1000
    :return:numpy ndarray representing the PPI interaction matrix between genes
    """
    gene_set = set(gene_list)
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}

    # Create an identity matrix as the initial PPI matrix
    n = len(gene_list)
    ppi_matrix = np.eye(n, dtype=np.float32)

    # Define chunk size for reading large files
    chunk_size = 1000000

    # Calculate total number of lines for progress tracking
    total_lines = sum(1 for _ in open(ppi_file_path, 'r'))

    dtype_dict = {
        'protein1': str,
        'protein2': str,
        'combined_score': np.float32
    }

    # Read the PPI file in chunks and filter for relevant genes
    for chunk in pd.read_csv(ppi_file_path,
                             sep='\t',
                             chunksize=chunk_size,
                             dtype=dtype_dict,
                             usecols=['protein1', 'protein2', 'combined_score']):

        # Use vectorized operations for gene filtering
        mask = (chunk['protein1'].isin(gene_set)) & (chunk['protein2'].isin(gene_set))
        filtered_chunk = chunk[mask]

        if not filtered_chunk.empty:
            idx1 = [gene_to_index[p] for p in filtered_chunk['protein1']]
            idx2 = [gene_to_index[p] for p in filtered_chunk['protein2']]

            if one_hot:
                ppi_matrix[idx1, idx2] = 1
                ppi_matrix[idx2, idx1] = 1  # Ensure symmetry
            else:
                ppi_matrix[idx1, idx2] = filtered_chunk['combined_score'].values / 1000
                ppi_matrix[idx2, idx1] = filtered_chunk['combined_score'].values / 1000

    print('PPI matrix shape:', ppi_matrix.shape)

    return ppi_matrix


def read_gene_set(gene_set_file_path: Union[str, List[str]],
                  min_n_genes: int = 15,
                  max_n_genes: int = 300,
                  max_overlap_ratio: float = 1.0) -> pd.DataFrame:
    """
    Read gene set files (GMT format) and create a gene-pathway matrix.

    :param gene_set_file_path: str or list of str, path(s) to GMT file(s)
    :param min_n_genes: int, minimum number of genes
    :param max_n_genes: int, maximum number of genes
    :param max_overlap_ratio: float, maximum overlap ratio
    :return pandas.DataFrame: gene-pathway matrix with genes as rows, pathways as columns, and values as 0 or 1
    """
    if isinstance(gene_set_file_path, str):
        gene_set_file_path = [gene_set_file_path]

    # Read gene set files
    gs2genes = {}
    all_genes = set()

    for gs_file in gene_set_file_path:
        if not os.path.exists(gs_file):
            raise FileNotFoundError(f'gene set file {gs_file} not found')
        with open(gs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) < 3:  # Ensure there are at least 3 columns
                        continue

                    gs = parts[0]  # Pathway name
                    genes = parts[2:]  # From the third column onwards are gene names

                    gs2genes[gs] = genes
                    all_genes.update(genes)

    # Filter gene sets based on criteria
    filtered_gs2genes = {}

    # Create a mapping from gene to pathways for overlap calculation
    gene_to_pathways = {}
    for gs, genes in gs2genes.items():
        for gene in genes:
            if gene not in gene_to_pathways:
                gene_to_pathways[gene] = set()
            gene_to_pathways[gene].add(gs)

    # Filter gene sets
    for gs, genes in gs2genes.items():
        gene_count = len(genes)
        genes_set = set(genes)

        # Condition 1: Gene count within specified range
        condition1 = min_n_genes <= gene_count <= max_n_genes

        # Condition 2: Gene count exceeds max_n_genes and overlap ratio is below threshold
        condition2 = False
        if gene_count > max_n_genes:
            # Calculate overlap ratio
            overlap_count = 0
            for gene in genes_set:
                if len(gene_to_pathways.get(gene, set()) - {gs}) > 0:
                    overlap_count += 1
            overlap_ratio = overlap_count / gene_count
            condition2 = overlap_ratio < max_overlap_ratio

        if condition1 or condition2:
            filtered_gs2genes[gs] = genes

    # Create gene-pathway matrix
    gene_set_df = pd.DataFrame(0, index=list(all_genes), columns=list(filtered_gs2genes.keys()))

    for gs, genes in filtered_gs2genes.items():
        gene_set_df.loc[genes, gs] = 1

    # Remove genes that are not in any pathway
    gene_set_df = gene_set_df.loc[gene_set_df.sum(axis=1) > 0]

    return gene_set_df


def get_pathway(gene_list, pathway_mask: pd.DataFrame):
    """
    Get the pathway mask for the specified gene list.

    :param gene_list: input gene list
    :param pathway_mask: pathway mask
    :return: pathway mask with genes in gene_list
    """
    common_genes = list(set(gene_list) & set(pathway_mask.index))
    print('Common genes between training set and pathway mask:', len(common_genes))
    genes_only_in_x = list(set(gene_list) - set(pathway_mask.index))
    # add genes only in x to pathway mask as all zeros
    if len(genes_only_in_x) > 0:
        print('Genes only in training set:', len(genes_only_in_x))
        pathway_mask = pd.concat([pathway_mask,
                                  pd.DataFrame(np.zeros((len(genes_only_in_x), pathway_mask.shape[1])),
                                               index=genes_only_in_x, columns=pathway_mask.columns)])
    pathway_mask = pathway_mask.loc[gene_list, :]  # genes by pathways
    print('pathway mask shape:', pathway_mask.shape)
    return pathway_mask


def load_knowledge(gene_list, is_ppi=True, is_pathway=True):
    """
    Load biological knowledge (PPI and/or pathway) for the specified gene list.

    :param gene_list: input gene list
    :param is_ppi: whether to load PPI data
    :param is_pathway: whether to load pathway data
    :return: knowledge matrix and knowledge names
    """
    if is_ppi:
        ppi_file_path = './data/knowledge/PPI_data_min700.txt'
        ppi = get_ppi(ppi_file_path=ppi_file_path, gene_list=gene_list, one_hot=False)

    if is_pathway:
        gmt_root_path = './data/knowledge/'
        gmt_list = ['c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt', 'c2.cp.reactome.v2024.1.Hs.symbols.gmt',
                    'c2.cp.pid.v2024.1.Hs.symbols.gmt', 'c2.cp.biocarta.v2024.1.Hs.symbols.gmt']
        gene_set_file_path = [os.path.join(gmt_root_path, gmt) for gmt in gmt_list]
        gene_set_df = read_gene_set(gene_set_file_path)
        pathways = get_pathway(gene_list, gene_set_df)
        pathways = pathways.values

    if is_ppi == True and is_pathway == True:
        knowledge = np.concatenate([ppi, pathways], axis=1)
    elif is_ppi:
        knowledge = ppi
    elif is_pathway:
        knowledge = pathways
    else:
        raise ValueError("At least one of ppi or pathway must be True.")
    knowledge_names = gene_list + gene_set_df.columns.to_list()

    return knowledge, knowledge_names
