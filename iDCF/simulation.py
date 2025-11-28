# -*- coding: UTF-8 -*-
"""
@Project: iDCF
@File   : train.py
@IDE    : PyCharm
@Author : hjguo
@Date   : 2025/7/9 11:37
@Doc    : Simulate pseudo-bulk data from single-cell data
"""
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random import choice
from typing import Union, Optional, Tuple


def _load_and_prepare_sc_data(sc_data: Union[str, pd.DataFrame, anndata.AnnData]) -> pd.DataFrame:
    """
    Load single-cell data.

    This function returns a standard pandas DataFrame regardless of the input format,
    with a numerical index, genes as columns, and a column named 'celltype'.

    :param sc_data: Input single-cell data, can be a file path, DataFrame, or AnnData object.
    :return: A processed pandas DataFrame.
    """
    print('Loading and preparing single-cell data...')

    if isinstance(sc_data, pd.DataFrame):
        # The data is already a DataFrame, no action needed.
        pass
    elif isinstance(sc_data, anndata.AnnData):
        print('Input is an AnnData object. Ensure "CellType" is in adata.obs.')
        if not isinstance(sc_data.X, np.ndarray):
            # Convert sparse matrix to dense array if needed.
            sc_data.X = sc_data.X.toarray()

        sc_data = pd.DataFrame(sc_data.X, index=sc_data.obs["CellType"], columns=sc_data.var.index)
    elif isinstance(sc_data, str):
        if '.txt' in sc_data:
            sc_data = pd.read_csv(sc_data, index_col=0, sep='\t')
        elif '.h5ad' in sc_data:
            print('You are using H5AD format data, please make sure "CellType" occurs in the adata.obs')
            sc_data = anndata.read_h5ad(sc_data)
            if not isinstance(sc_data.X, np.ndarray):
                # Convert sparse matrix to dense array if needed.
                sc_data.X = sc_data.X.toarray()

            sc_data = pd.DataFrame(sc_data.X, index=sc_data.obs["CellType"], columns=sc_data.var.index)
    else:
        raise TypeError("sc_data must be a string, pandas.DataFrame, or anndata.AnnData object.")

    sc_data.dropna(inplace=True)
    sc_data['celltype'] = sc_data.index
    sc_data.reset_index(drop=True, inplace=True)
    print('Reading dataset is done.')
    return sc_data


def simulation(
        sc_data: Union[str, pd.DataFrame, anndata.AnnData],
        out_name: Optional[str] = None,
        total_cells_per_sample: int = 500,
        num_samples: int = 8000,
        distribution_function: str = 'dirichlet',
        sparse: bool = True,
        sparse_prob: float = 0.5,
        cell_count_variation: bool = True,
        cell_count_variation_range: Tuple[float, float] = (0.8, 1.2),
        add_noise: bool = False
) -> anndata.AnnData:
    """
    Generates simulated pseudo-bulk data based on a single-cell reference.

    :param sc_data: Single-cell data, can be a file path (str), pandas DataFrame, or anndata.AnnData object.
                    For a DataFrame, the index should be cell types. For an AnnData object, cell types should be in `adata.obs['CellType']`.
    :param out_name: Output filename (without extension). If provided, the data will be saved as an .h5ad file.
    :param total_cells_per_sample: Total number of cells in each simulated sample.
    :param num_samples: Number of simulated samples to generate.
    :param distribution_function: Distribution function to generate cell proportions ('dirichlet' or 'uniform').
    :param sparse: Whether to make the cell composition of some samples sparse.
    :param sparse_prob: If sparse=True, this controls the degree of sparsity and the proportion of affected samples.
    :param cell_count_variation: Whether to introduce Poisson-distributed variation in cell counts.
    :param cell_count_variation_range: If cell_count_variation=True, the range for cell count fluctuation.
    :param add_noise: Whether to add Gaussian noise to the final pseudo-bulk data.
    :return: An anndata.AnnData object containing the simulated data and corresponding cell proportions.
    """
    sc_data_df = _load_and_prepare_sc_data(sc_data)

    num_celltypes = len(sc_data_df['celltype'].value_counts())
    gene_names = sc_data_df.columns[:-1]
    celltype_groups = sc_data_df.groupby('celltype').groups
    sc_data_df.drop(columns='celltype', inplace=True)

    sc_expression_matrix = sc_data_df.to_numpy()
    # Use C-contiguous array for better performance.
    sc_expression_matrix = np.ascontiguousarray(sc_expression_matrix, dtype=np.float32)

    # Generate random cell proportions.
    if distribution_function == 'dirichlet':
        print(f'Using Dirichlet distribution to generate {num_samples} samples...')
        proportions = np.random.dirichlet(np.ones(num_celltypes), num_samples)
    elif distribution_function == 'uniform':
        print(f'Using uniform distribution to generate {num_samples} samples...')
        proportions = np.random.uniform(0, 1, (num_samples, num_celltypes))
    else:
        raise ValueError("The distribution function must be 'dirichlet' or 'uniform'.")

    # Convert group indices to numpy arrays for faster access.
    for key, value in celltype_groups.items():
        celltype_groups[key] = np.array(value)

    proportions = proportions / np.sum(proportions, axis=1).reshape(-1, 1)

    # Make some cell fractions sparse if specified.
    if sparse:
        print(f"Sparsifying cell fractions with probability {sparse_prob}...")
        # Make a subset of samples sparse.
        for i in range(int(proportions.shape[0] * sparse_prob)):
            indices = np.random.choice(np.arange(proportions.shape[1]), replace=False,
                                       size=int(proportions.shape[1] * sparse_prob))
            proportions[i, indices] = 0
        proportions = proportions / np.sum(proportions, axis=1).reshape(-1, 1)

    # Calculate the number of cells of each type for each sample.
    if cell_count_variation:
        expected_cells = total_cells_per_sample * proportions
        cell_count_variation = np.random.poisson(lam=expected_cells).astype(int)

        # Calculate the allowed fluctuation range.
        min_val = np.floor(expected_cells * cell_count_variation_range[0]).astype(int)
        max_val = np.floor(expected_cells * cell_count_variation_range[1]).astype(int)

        # Apply constraints to prevent extreme values.
        cell_counts = np.clip(cell_count_variation, min_val, max_val)
    else:
        cell_counts = np.floor(total_cells_per_sample * proportions).astype(int)

    # Calculate precise proportions based on the final cell counts.
    proportions = cell_counts / np.sum(cell_counts, axis=1).reshape(-1, 1)

    # Start sampling to generate pseudo-bulk data.
    pseudo_bulk_samples = np.zeros((proportions.shape[0], sc_expression_matrix.shape[1]))
    all_celltypes = list(celltype_groups.keys())
    print('Sampling cells to compose pseudo-bulk data...')
    for i, sample_cell_counts in tqdm(enumerate(cell_counts), total=num_samples):
        for j, cell_name in enumerate(all_celltypes):
            count = int(sample_cell_counts[j])
            if count > 0:
                selected_indices = choice(celltype_groups[cell_name], size=count, replace=True)
                pseudo_bulk_samples[i] += sc_expression_matrix[selected_indices].sum(axis=0)

    # Add noise if specified.
    if add_noise:
        noise = np.random.normal(loc=0, scale=0.1, size=pseudo_bulk_samples.shape)
        pseudo_bulk_samples += noise

    proportions_df = pd.DataFrame(proportions, columns=all_celltypes)
    simulated_data = anndata.AnnData(X=pseudo_bulk_samples,
                                     obs=proportions_df,
                                     var=pd.DataFrame(index=gene_names))

    print('Simulation is done.')
    if out_name is not None:
        output_path = out_name + '.h5ad'
        simulated_data.write_h5ad(output_path)
        print(f"Saved simulated data to {output_path}")

    return simulated_data


if __name__ == '__main__':
    # PBMC
    data = pd.read_csv('../data/scRNA-seq/PBMC/pbmc8k/data8k_norm_counts_all.txt', index_col=0, sep='\t')
    # celltype = pd.read_csv('../data/scRNA-seq/PBMC/pbmc8k/data8k_celltypes.txt', index_col=0, sep='\t')
    celltype = pd.read_csv('../data/scRNA-seq/PBMC/pbmc8k/CellTypist_predicted_celltypes.txt', index_col=0, sep='\t')
    celltype.replace({'Dendritics': 'Unknown'}, inplace=True)
    data.index = celltype['Celltype']
    out_name = '../data/Simu bulk RNA-seq/new_data8k_norm_count_my_500cell_simu'

    # --- Run simulation ---
    simulation(
        sc_data=data,
        out_name=out_name,
        total_cells_per_sample=500,
        num_samples=8000
    )
