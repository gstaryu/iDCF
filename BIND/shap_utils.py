# -*- coding: UTF-8 -*-
"""
@Project: MyDeconv
@File   : shap_utils.py
@IDE    : PyCharm
@Author : staryu
@Date   : 2025/7/11 14:51
@Doc    : 除了官方文档外，可以参考：https://zhuanlan.zhihu.com/p/701713976
"""
import shap
import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def explain_expression_genes(model, data_tensor, feature_names, save_path):
    print("[SHAP] 分析只有基因表达情况下输入基因重要性...")
    model.eval()

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x, is_knowledge=False)

    wrapped_model = WrappedModel(model).to(model.device)
    # background = data_tensor[:50].to(model.device)  # 选前50个样本作为背景，SHAP需要背景样本来计算基线影响，论文中的结果应该取30-100
    background = data_tensor.to(model.device)
    explainer = shap.GradientExplainer(wrapped_model, background)

    # shap_values = explainer.shap_values(data_tensor[:50].to(model.device))
    shap_values = explainer.shap_values(data_tensor.to(model.device))

    for i, label in enumerate(model.labels):
        # shap.summary_plot(shap_values[:, :, i], data_tensor[:50].cpu().numpy(), feature_names=feature_names, show=False, max_display=20)
        shap.summary_plot(shap_values[:, :, i], data_tensor.cpu().numpy(), feature_names=feature_names, show=False,
                          max_display=20)
        plt.title(f"SHAP for cell type: {label}")
        plt.savefig(os.path.join(save_path, f"shap_expression_by_gene_{label}.png"))
        plt.close()

    np.save(os.path.join(save_path, f"shap_expression_by_gene_values.npy"), shap_values)
    return shap_values


def explain_knowledge_genes(model, data_tensor, feature_names, save_path):
    print("[SHAP] 分析加入知识之后输入基因重要性...")
    model.eval()

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x, is_knowledge=True)

    wrapped_model = WrappedModel(model).to(model.device)
    # background = data_tensor[:50].to(model.device)
    background = data_tensor.to(model.device)
    explainer = shap.GradientExplainer(wrapped_model, background)

    # shap_values = explainer.shap_values(data_tensor[:50].to(model.device))
    shap_values = explainer.shap_values(data_tensor.to(model.device))

    for i, label in enumerate(model.labels):
        # shap.summary_plot(shap_values[:, :, i], data_tensor[:50].cpu().numpy(), feature_names=feature_names, show=False, max_display=20)
        shap.summary_plot(shap_values[:, :, i], data_tensor.cpu().numpy(), feature_names=feature_names, show=False,
                          max_display=20)
        plt.title(f"SHAP for cell type: {label}")
        plt.savefig(os.path.join(save_path, f"shap_knowledge_by_gene_{label}.png"))
        plt.close()

    np.save(os.path.join(save_path, f"shap_knowledge_by_gene_values.npy"), shap_values)
    return shap_values


def explain_branch_contributions(model, data_tensor, save_path):
    print("[SHAP] 分析知识分支 vs 原始分支贡献...")
    model.eval()

    class WrappedBranch(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out1 = self.model.DNN(x)  # (N, 32)
            out2 = self.model.KSNN(x)  # (N, 32)
            combined = torch.cat([out1, out2], dim=1)  # (N, 64)
            return combined

    wrapped_model = WrappedBranch(model).to(model.device)
    background = data_tensor[:50].to(model.device)
    # background = data_tensor.to(model.device)
    explainer = shap.GradientExplainer(wrapped_model, background)

    # SHAP values: (samples, input_dim, 64)
    shap_values = explainer.shap_values(data_tensor[:50].to(model.device))
    # shap_values = explainer.shap_values(data_tensor.to(model.device))

    shap_values_arr = shap_values  # (samples, input_dim, 64)

    # 平均所有样本
    shap_mean_input_to_branch = np.mean(np.abs(shap_values_arr), axis=0)  # (input_dim, 64)

    # raw 分支是前32维，knowledge 分支是后32维
    shap_mlp1 = shap_mean_input_to_branch[:, :32]  # (input_dim, 32)
    shap_mlp2 = shap_mean_input_to_branch[:, 32:]  # (input_dim, 32)

    # 每条分支的总重要性：对每个输入特征取总贡献
    total_mlp1 = shap_mlp1.sum()
    total_mlp2 = shap_mlp2.sum()

    # 归一化
    total_mlp1 /= shap_mean_input_to_branch.sum()
    total_mlp2 /= shap_mean_input_to_branch.sum()

    print(f"Total SHAP contribution - raw: {total_mlp1:.4f}, knowledge: {total_mlp2:.4f}")

    # 可视化：画柱状图
    plt.figure(figsize=(6, 5))
    plt.bar(['Raw Input Branch', 'Knowledge Branch'], [total_mlp1, total_mlp2], color=['skyblue', 'orange'])
    plt.ylabel("Total SHAP Contribution")
    plt.title("Branch-wise Contribution to Representation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "shap_branch_contribution_bar.png"))
    plt.close()

    # 可选：保存 SHAP 数值
    np.save(os.path.join(save_path, "shap_branch_contribution_values.npy"), shap_values_arr)

    return shap_values_arr


# 计算均值的方法
def explain_knowledge_features(model, data_tensor, knowledge_feature_names, save_path):
    print("[SHAP] 分析知识项（PPI/Pathways）重要性...")
    # model = model.model[0]
    model.eval()

    # 计算知识特征向量（即PPI+Pathway特征）
    knowledge_tensor = model.KSNN[0](data_tensor.to(model.device)).detach()

    class KnowledgeOnlyModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, k):
            return self.model.KSNN[1:](k)

    wrapped_model = KnowledgeOnlyModel(model).to(model.device)

    # 用前10个knowledge向量作为背景
    # background = knowledge_tensor[:50]
    background = knowledge_tensor
    explainer = shap.GradientExplainer(wrapped_model, background)

    # 计算SHAP值
    # shap_values = explainer.shap_values(knowledge_tensor[:50])  # shape: (samples, 15560, C)
    shap_values = explainer.shap_values(knowledge_tensor)  # shape: (samples, 15560, C)

    # 取绝对值后对输出维度求均值（不保留正负方向）
    shap_vals_mean = np.mean(np.abs(shap_values), axis=2)  # shape: (10, 15560)

    # shap.summary_plot(shap_vals_mean, knowledge_tensor[:50].cpu().numpy(), feature_names=knowledge_feature_names, show=False, max_display=30)
    shap.summary_plot(shap_vals_mean, knowledge_tensor.cpu().numpy(), feature_names=knowledge_feature_names, show=False,
                      max_display=30)

    plt.gcf().set_size_inches(18, 10)
    plt.tight_layout()
    # plt.show(bbox_inches='tight')
    # plt.savefig(os.path.join(save_path, "shap_knowledge_features.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(save_path, "shap_knowledge_features.png"), bbox_inches='tight', dpi=300)
    np.save(os.path.join(save_path, "knowledge_tensor.npy"), knowledge_tensor.cpu().numpy())
    np.save(os.path.join(save_path, "shap_knowledge_values.npy"), shap_values)
    plt.close()

    return shap_values
