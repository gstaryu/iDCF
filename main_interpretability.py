# -*- coding: UTF-8 -*-
"""
@Project: BIND
@File   : main.py.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2025/7/9 11:35
@Doc    : 
"""
import os
import torch
import numpy as np
import pandas as pd
import random
from BIND.utils import load_data, load_knowledge
from BIND.train import train, prediction
from BIND.shap_utils import (
    explain_expression_genes,
    explain_knowledge_genes,
    explain_branch_contributions,
    explain_knowledge_features
)

def main(data_name=None, run_epochs=3, is_knowledge=True, interpretability=False):
    print(f'Data_name: {data_name}, is_knowledge: {is_knowledge}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = {'batch_size': 64, 'lr': 0.0001, 'epochs': 128}
    save_path = './results/'

    # load data
    print("------Load Data------")
    train_data, test_x = load_data(data_name=data_name)

    gene_list = test_x.var_names.tolist()
    if is_knowledge:
        knowledge = load_knowledge(gene_list)

    pred_list = []
    for i in range(run_epochs):
        print(f"------Run epochs {i}------")
        model, loss = train(train_data=train_data, batch_size=params['batch_size'], lr=params['lr'], epochs=params['epochs'], device=device,
                            knowledge=knowledge if is_knowledge else None)
        pred = prediction(model=model, test_data=test_x, device=device,
                          knowledge=knowledge if is_knowledge else None)
        pred.to_csv(os.path.join(save_path, f"{data_name}_pred_{i}.txt"), sep='\t')
        pred_list.append(pred)
    final_pred = pd.concat(pred_list)
    final_pred = final_pred.groupby(level=0, sort=False).mean()
    final_pred.to_csv(os.path.join(save_path, f"{data_name}_pred.txt"), sep='\t')

    if interpretability:
        interpretability_path = os.path.join(save_path, f'interpretability/{data_name}')
        idx = random.sample(range(train_data.shape[0]), 1000)
        X_tensor = torch.tensor(train_data.X[idx], dtype=torch.float32)
        gene_names = gene_list  # 输入基因名

        if is_knowledge:
            knowledge_names = knowledge.shape
        else:
            knowledge_names = []

        with open(os.path.join(interpretability_path, "gene_names.txt"), "w") as f:
            for name in gene_names:
                f.write(f"{name}\n")

        with open(os.path.join(interpretability_path, "knowledge_names.txt"), "w") as f:
            for name in knowledge_names:
                f.write(f"{name}\n")

        # 保存X_tensor为.npy文件
        np.save(os.path.join(interpretability_path, "X_tensor.npy"), X_tensor.cpu().numpy())

        # === 运行 SHAP 分析 ===
        explain_expression_genes(model, X_tensor, gene_names, interpretability_path)

        if is_knowledge:
            explain_knowledge_genes(model, X_tensor, gene_names, interpretability_path)
            # explain_branch_contributions(model, X_tensor, interpretability_path)
            explain_knowledge_features(model, X_tensor, knowledge_names, interpretability_path)


if __name__ == "__main__":
    main(data_name='monaco_pbmc', run_epochs=3, is_knowledge=True)
    # main(data_name='sdy67', run_epochs=3, is_knowledge=True)
    # main(data_name='microarray', run_epochs=3, is_knowledge=True)
    # main(data_name='brain_human', run_epochs=3, is_knowledge=True)
    # main(data_name='GSE107572', run_epochs=3, is_knowledge=True)
    # main(data_name='GSE120502', run_epochs=3, is_knowledge=True)
    # main(data_name='monaco2', run_epochs=3, is_knowledge=True)
    # main(data_name='sdy67_250', run_epochs=3, is_knowledge=True)
    # main(data_name='GSE193141', run_epochs=3, is_knowledge=True)
    # # main(data_name='HNSC', run_epochs=3, is_knowledge=True)
    # main(data_name='Islet', run_epochs=3, is_knowledge=True)
    # main(data_name='Pancreas', run_epochs=3, is_knowledge=True)
    # main(data_name='CRC-sEV', run_epochs=3, is_knowledge=True)
    # main(data_name='HGSC', run_epochs=3, is_knowledge=True)