# -*- coding: UTF-8 -*-
"""
@Project: BIND
@File   : utils.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2025/7/9 11:37
@Doc    : 包括loss, data_load, 转化函数等
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
        # 转换为tpm
        # data_df_tpm = count2tpm(data.to_df().T, annotation_file_path='./data/gencode.gene.info.v22.tsv')
        # data = ad.AnnData(X=data_df_tpm.values, obs=data.obs, var=pd.DataFrame(index=data_df_tpm.columns))
        # 转换为CPM
        # sc.pp.normalize_total(data, target_sum=1e6)
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

    # # 测试集CPM标准化
    # ad_test_x = ad.AnnData(X=test_x.values, var=pd.DataFrame(index=test_x.columns.tolist()))
    # sc.pp.normalize_total(ad_test_x, target_sum=1e6)
    # test_x = pd.DataFrame(ad_test_x.X, index=test_x.index.tolist(), columns=test_x.columns.tolist())

    print(f'Original train_x shape: {train_x.shape}, test_x shape: {test_x.shape}')

    print('Cutting variance...')
    # 删除全0基因
    train_x = train_x.loc[:, (train_x != 0).any(axis=0)]  # 删除全0基因
    test_x = test_x.loc[:, (test_x != 0).any(axis=0)]  # 删除全0基因
    # variance_threshold = 0.99999
    variance_threshold = 0.99
    var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[
        int(train_x.shape[1] * variance_threshold)]  # 计算每个基因的方差，从大到小排序，找到排名在前variance_threshold的基因
    train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]  # 这里是把方差小于var_cutoff的基因去掉
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

    # 转成AnnData
    train_data = ad.AnnData(X=train_x.values, obs=train_data.obs)
    train_data.var_names = train_x.columns
    train_data.uns = data.uns

    return train_data, test_data


def load_data_from_path(train_path, test_path):
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
    # 删除全0基因
    train_x = train_x.loc[:, (train_x != 0).any(axis=0)]  # 删除全0基因
    test_x = test_x.loc[:, (test_x != 0).any(axis=0)]  # 删除全0基因
    # variance_threshold = 0.99999
    variance_threshold = 0.99
    var_cutoff = train_x.var(axis=0).sort_values(ascending=False)[
        int(train_x.shape[1] * variance_threshold)]  # 计算每个基因的方差，从大到小排序，找到排名在前variance_threshold的基因
    train_x = train_x.loc[:, train_x.var(axis=0) > var_cutoff]  # 这里是把方差小于var_cutoff的基因去掉
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

    # 转成AnnData
    train_data = ad.AnnData(X=train_x.values, obs=train_data.obs)
    train_data.var_names = train_x.columns
    train_data.uns = data.uns

    return train_data, test_data


def get_ppi(ppi_file_path, gene_list, one_hot=True):
    """
    从PPI数据文件中提取指定基因列表的蛋白质互作网络，优化版本

    参数:
    gene_list (list): 基因名称列表

    返回:
    numpy.ndarray: 基因间PPI互作矩阵
    """
    # 将基因列表转换为集合以加速查找
    gene_set = set(gene_list)
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}

    # 创建初始矩阵
    n = len(gene_list)
    ppi_matrix = np.eye(n, dtype=np.float32)

    # 使用分块读取来处理大文件
    chunk_size = 1000000  # 每次读取100万行

    # 首先计算总行数
    total_lines = sum(1 for _ in open(ppi_file_path, 'r'))

    # 使用dtype和usecols来只读取需要的列
    dtype_dict = {
        'protein1': str,
        'protein2': str,
        'combined_score': np.float32
    }

    # 使用迭代器分块读取文件，进度条基于总行数
    for chunk in pd.read_csv(ppi_file_path,
                             sep='\t',
                             chunksize=chunk_size,
                             dtype=dtype_dict,
                             usecols=['protein1', 'protein2', 'combined_score']):

        # 使用向量化操作进行基因筛选
        mask = (chunk['protein1'].isin(gene_set)) & (chunk['protein2'].isin(gene_set))
        filtered_chunk = chunk[mask]

        # 批量更新矩阵
        if not filtered_chunk.empty:
            idx1 = [gene_to_index[p] for p in filtered_chunk['protein1']]
            idx2 = [gene_to_index[p] for p in filtered_chunk['protein2']]

            # 批量赋值
            if one_hot:
                ppi_matrix[idx1, idx2] = 1
                ppi_matrix[idx2, idx1] = 1  # 确保矩阵对称
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
    读取基因集合文件(GMT格式)并创建基因-通路矩阵

    参数:
    gene_set_file_path: str或str列表，GMT文件路径
    min_n_genes: int，最小基因数量
    max_n_genes: int，最大基因数量
    max_overlap_ratio: float，最大重叠比例

    返回:
    pandas.DataFrame: 基因-通路矩阵，行为基因，列为通路，值为0或1
    """
    # 确保file_path是列表
    if isinstance(gene_set_file_path, str):
        gene_set_file_path = [gene_set_file_path]

    # 读取所有基因集
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
                    if len(parts) < 3:  # 确保至少有通路名称、描述和一个基因
                        continue

                    gs = parts[0]  # 通路名称
                    genes = parts[2:]  # 从第三列开始都是基因

                    # 应用基因数量上限
                    # if len(genes) > max_n_genes:
                    #     genes = genes[:max_n_genes]

                    gs2genes[gs] = genes
                    all_genes.update(genes)

    # 过滤基因集（修复版本）
    filtered_gs2genes = {}

    # 预处理：构建基因到通路的反向索引
    gene_to_pathways = {}
    for gs, genes in gs2genes.items():
        for gene in genes:  # 去重
            if gene not in gene_to_pathways:
                gene_to_pathways[gene] = set()
            gene_to_pathways[gene].add(gs)

    # 主逻辑
    for gs, genes in gs2genes.items():
        gene_count = len(genes)
        genes_set = set(genes)  # 去重后的基因集合

        # 条件1: 基因数范围
        condition1 = min_n_genes <= gene_count <= max_n_genes

        # 条件2: 大通路且低重叠率
        condition2 = False
        if gene_count > max_n_genes:
            # 正确计算重叠率
            overlap_count = 0
            for gene in genes_set:
                if len(gene_to_pathways.get(gene, set()) - {gs}) > 0:
                    overlap_count += 1
            overlap_ratio = overlap_count / gene_count
            condition2 = overlap_ratio < max_overlap_ratio

        # 保留条件
        if condition1 or condition2:
            filtered_gs2genes[gs] = genes

    # 创建基因-通路矩阵
    gene_set_df = pd.DataFrame(0, index=list(all_genes), columns=list(filtered_gs2genes.keys()))

    for gs, genes in filtered_gs2genes.items():
        # 使用loc为特定基因和通路设置值为1
        gene_set_df.loc[genes, gs] = 1

    # 将行和为0的行删除
    gene_set_df = gene_set_df.loc[gene_set_df.sum(axis=1) > 0]

    return gene_set_df


def get_pathway(gene_list, pathway_mask: pd.DataFrame):
    """
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

    return knowledge
