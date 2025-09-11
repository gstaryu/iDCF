# -*- coding: UTF-8 -*-
"""
@Project: BIND
@File   : main.py
@IDE    : PyCharm
@Author : hjguo
@Date   : 2025/7/9 11:35
@Doc    : Main function
"""
import os
import torch
import pandas as pd
from BIND.utils import load_data, load_data_from_path, load_knowledge
from BIND.train import train, prediction


def main(data_name=None, run_epochs=3, is_knowledge=True):
    """
    Main function to train and predict using the BIND model.

    :param data_name: Name of the dataset to be used.
    :param run_epochs: Number of times to run the training and prediction process.
    :param is_knowledge: Boolean indicating whether to incorporate prior knowledge into the model.
    :return: None
    """
    print(f'Data_name: {data_name}, is_knowledge: {is_knowledge}')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = {'batch_size': 64, 'lr': 0.0001, 'epochs': 128}
    save_path = './results/'

    # load data
    print("------Load Data------")
    train_data, test_x = load_data(data_name=data_name)

    if is_knowledge:
        gene_list = test_x.var_names.tolist()
        knowledge, _ = load_knowledge(gene_list)

    pred_list = []
    for i in range(run_epochs):
        print(f"------Run epochs {i}------")
        model, loss = train(train_data=train_data, batch_size=params['batch_size'], lr=params['lr'],
                            epochs=params['epochs'], device=device,
                            knowledge=knowledge if is_knowledge else None)
        pred = prediction(model=model, test_data=test_x, device=device,
                          knowledge=knowledge if is_knowledge else None)
        pred.to_csv(os.path.join(save_path, f"{data_name}_pred_{i}.txt"), sep='\t')
        pred_list.append(pred)
    final_pred = pd.concat(pred_list)
    final_pred = final_pred.groupby(level=0, sort=False).mean()
    final_pred.to_csv(os.path.join(save_path, f"{data_name}_pred.txt"), sep='\t')


if __name__ == "__main__":
    # main(data_name='monaco_pbmc', run_epochs=3, is_knowledge=True)
    # main(data_name='sdy67', run_epochs=3, is_knowledge=True)
    # main(data_name='microarray', run_epochs=3, is_knowledge=True)
    main(data_name='brain_human', run_epochs=3, is_knowledge=True)
    main(data_name='GSE107572', run_epochs=3, is_knowledge=True)
    main(data_name='GSE120502', run_epochs=3, is_knowledge=True)
    main(data_name='monaco2', run_epochs=3, is_knowledge=True)
    main(data_name='sdy67_250', run_epochs=3, is_knowledge=True)
    main(data_name='GSE193141', run_epochs=3, is_knowledge=True)
    # main(data_name='HNSC', run_epochs=3, is_knowledge=True)
    main(data_name='Islet', run_epochs=3, is_knowledge=True)
    main(data_name='Pancreas', run_epochs=3, is_knowledge=True)
    main(data_name='CRC-sEV', run_epochs=3, is_knowledge=True)
    main(data_name='HGSC', run_epochs=3, is_knowledge=True)
