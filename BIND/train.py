# -*- coding: UTF-8 -*-
"""
@Project: BIND
@File   : train.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2025/7/9 11:37
@Doc    : 训练和测试代码
"""
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from .model import BIND

from torch.utils.data import DataLoader

class Dataset:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def train(train_data, batch_size, lr, epochs, device, knowledge=None):
    X_tensor, Y_tensor = torch.from_numpy(train_data.X).float(), torch.from_numpy(train_data.obs.to_numpy()).float()
    X_tensor, Y_tensor = X_tensor.to(device), Y_tensor.to(device)
    train_loader = DataLoader(Dataset(X_tensor, Y_tensor), batch_size=batch_size, shuffle=True)
    if knowledge is not None:
        knowledge = torch.tensor(knowledge)
    is_knowledge = True if knowledge is not None else False
    model = BIND(gene_num=train_data.X.shape[1], output_dim=train_data.obs.shape[1], knowledge=knowledge).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss = []

    pbar = tqdm(range(epochs), desc='Training')
    for epoch in pbar:
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            frac_pred = model(batch_x, is_knowledge=is_knowledge)
            batch_loss = nn.MSELoss()(frac_pred, batch_y) + nn.L1Loss()(frac_pred, batch_y)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        avg_loss = total_loss / len(train_loader)
        loss.append(avg_loss)
        pbar.set_postfix(Loss=f'{avg_loss:.4f}')
    return model, loss

def prediction(model, test_data, device, knowledge=None):
    model.eval()
    is_knowledge = True if knowledge is not None else False
    X_tensor = torch.from_numpy(test_data.X).float().to(device)
    with torch.no_grad():
        frac_pred = model(X_tensor, is_knowledge=is_knowledge).cpu().numpy()
    test_pred = pd.DataFrame(frac_pred, index=test_data.obs_names, columns=test_data.uns['cell_types'])
    return test_pred
