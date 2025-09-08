# -*- coding: UTF-8 -*-
"""
@Project: BIND
@File   : dataset.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2025/7/9 11:36
@Doc    : 加载数据
"""
class Dataset:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


# 后面直接套用torch的DataLoader，data用simu.X, label用simu.obs