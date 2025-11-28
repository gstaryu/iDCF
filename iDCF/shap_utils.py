# -*- coding: UTF-8 -*-
"""
@Project: iDCF
@File   : shap_utils.py
@IDE    : PyCharm
@Author : hjguo
@Date   : 2025/7/11 14:51
@Doc    : SHAP interpretability analysis code
"""
import shap
import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def explain_expression_genes(model, data_tensor, feature_names, cell_types, save_path):
    """
    Analyze the contribution of each gene when only gene expression data is input.

    :param model: The trained iDCF model.
    :param data_tensor: Input gene expression data as a tensor.
    :param feature_names: List of gene names corresponding to the features in data_tensor.
    :param cell_types: List of cell type names corresponding to the output classes of the model.
    :param save_path: Directory to save the SHAP plots and values.
    :return: SHAP values array.
    """
    print("[SHAP] Analyzing the contribution of each gene when only gene expression data is input...")
    device = next(model.parameters()).device
    model.eval()

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x, is_knowledge=False)

    wrapped_model = WrappedModel(model).to(device)
    # background = data_tensor[:50].to(device)  # Select a subset as background
    background = data_tensor.to(device)
    explainer = shap.GradientExplainer(wrapped_model, background)

    # shap_values = explainer.shap_values(data_tensor[:50].to(device))
    shap_values = explainer.shap_values(data_tensor.to(device))

    for i, label in enumerate(cell_types):
        # shap.summary_plot(shap_values[:, :, i], data_tensor[:50].cpu().numpy(), feature_names=feature_names, show=False, max_display=20)
        shap.summary_plot(shap_values[:, :, i], data_tensor.cpu().numpy(), feature_names=feature_names, show=False,
                          max_display=20)
        plt.title(f"SHAP for cell type: {label}")
        plt.savefig(os.path.join(save_path, f"shap_expression_by_gene_{label}.png"))
        plt.close()

    np.save(os.path.join(save_path, f"shap_expression_by_gene_values.npy"), shap_values)
    return shap_values


def explain_knowledge_genes(model, data_tensor, feature_names, cell_types, save_path):
    """
    Analyze the contribution of each gene when incorporating prior knowledge (PPI/Pathways).

    :param model: The trained iDCF model.
    :param data_tensor: Input gene expression data as a tensor.
    :param feature_names: List of gene names corresponding to the features in data_tensor.
    :param cell_types: List of cell type names corresponding to the output classes of the model.
    :param save_path: Directory to save the SHAP plots and values.
    :return: SHAP values array.
    """
    print("[SHAP] Analyzing the contribution of each gene when incorporating knowledge...")
    device = next(model.parameters()).device
    model.eval()

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x, is_knowledge=True)

    wrapped_model = WrappedModel(model).to(device)
    # background = data_tensor[:50].to(device)
    background = data_tensor.to(device)
    explainer = shap.GradientExplainer(wrapped_model, background)

    # shap_values = explainer.shap_values(data_tensor[:50].to(device))
    shap_values = explainer.shap_values(data_tensor.to(device))

    for i, label in enumerate(cell_types):
        # shap.summary_plot(shap_values[:, :, i], data_tensor[:50].cpu().numpy(), feature_names=feature_names, show=False, max_display=20)
        shap.summary_plot(shap_values[:, :, i], data_tensor.cpu().numpy(), feature_names=feature_names, show=False,
                          max_display=20)
        plt.title(f"SHAP for cell type: {label}")
        plt.savefig(os.path.join(save_path, f"shap_knowledge_by_gene_{label}.png"))
        plt.close()

    np.save(os.path.join(save_path, f"shap_knowledge_by_gene_values.npy"), shap_values)
    return shap_values


def explain_branch_contributions(model, data_tensor, save_path):
    """
    Analyze the contributions of the knowledge branch and gene expression data branch.

    :param model: The trained iDCF model.
    :param data_tensor: Input gene expression data as a tensor.
    :param save_path: Directory to save the SHAP plots and values.
    :return: SHAP values array.
    """
    print("[SHAP] Analyzing knowledge branches and gene expression data branch contributions...")
    device = next(model.parameters()).device
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

    wrapped_model = WrappedBranch(model).to(device)
    background = data_tensor[:50].to(device)
    # background = data_tensor.to(device)
    explainer = shap.GradientExplainer(wrapped_model, background)

    # SHAP values: (samples, input_dim, 64)
    shap_values = explainer.shap_values(data_tensor[:50].to(device))
    # shap_values = explainer.shap_values(data_tensor.to(device))

    shap_values_arr = shap_values  # (samples, input_dim, 64)

    # Calculate mean absolute SHAP values across samples for each input feature to each branch
    shap_mean_input_to_branch = np.mean(np.abs(shap_values_arr), axis=0)  # (input_dim, 64)

    shap_mlp1 = shap_mean_input_to_branch[:, :32]  # (input_dim, 32)
    shap_mlp2 = shap_mean_input_to_branch[:, 32:]  # (input_dim, 32)

    total_mlp1 = shap_mlp1.sum()
    total_mlp2 = shap_mlp2.sum()

    total_mlp1 /= shap_mean_input_to_branch.sum()
    total_mlp2 /= shap_mean_input_to_branch.sum()

    print(f"Total SHAP contribution - raw: {total_mlp1:.4f}, knowledge: {total_mlp2:.4f}")

    plt.figure(figsize=(6, 5))
    plt.bar(['Raw Input Branch', 'Knowledge Branch'], [total_mlp1, total_mlp2], color=['skyblue', 'orange'])
    plt.ylabel("Total SHAP Contribution")
    plt.title("Branch-wise Contribution to Representation")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "shap_branch_contribution_bar.png"))
    plt.close()

    np.save(os.path.join(save_path, "shap_branch_contribution_values.npy"), shap_values_arr)

    return shap_values_arr


# 计算均值的方法
def explain_knowledge_features(model, data_tensor, knowledge_feature_names, save_path):
    """
    Analyze the contribution of knowledge items (PPI/Pathways).

    :param model: The trained iDCF model.
    :param data_tensor: Input gene expression data as a tensor.
    :param knowledge_feature_names: List of knowledge feature names corresponding to the knowledge items.
    :param save_path: Directory to save the SHAP plots and values.
    :return: SHAP values array.
    """
    print("[SHAP] Analyzing the contribution of knowledge items (PPI/Pathways)...")
    device = next(model.parameters()).device
    model.eval()

    # Calculate knowledge tensor
    knowledge_tensor = model.KSNN[0](data_tensor.to(device)).detach()

    class KnowledgeOnlyModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, k):
            return self.model.KSNN[1:](k)

    wrapped_model = KnowledgeOnlyModel(model).to(device)

    # background = knowledge_tensor[:50]  # Select a subset as background
    background = knowledge_tensor
    explainer = shap.GradientExplainer(wrapped_model, background)

    # Calculate SHAP values for knowledge tensor
    # shap_values = explainer.shap_values(knowledge_tensor[:50])  # shape: (samples, 15560, C)
    shap_values = explainer.shap_values(knowledge_tensor)  # shape: (samples, 15560, C)

    # Take the absolute value and then average over the output dimension (not retaining positive/negative direction)
    shap_vals_mean = np.mean(np.abs(shap_values), axis=2)  # shape: (1000, 15560)

    # shap.summary_plot(shap_vals_mean, knowledge_tensor[:50].cpu().numpy(), feature_names=knowledge_feature_names, show=False, max_display=30)
    shap.summary_plot(shap_vals_mean, knowledge_tensor.cpu().numpy(), feature_names=knowledge_feature_names, show=False,
                      max_display=30)

    plt.gcf().set_size_inches(18, 10)
    plt.tight_layout()
    # plt.show(bbox_inches='tight')
    plt.savefig(os.path.join(save_path, "shap_knowledge_features.png"), bbox_inches='tight', dpi=300)
    np.save(os.path.join(save_path, "knowledge_tensor.npy"), knowledge_tensor.cpu().numpy())
    np.save(os.path.join(save_path, "shap_knowledge_values.npy"), shap_values)
    plt.close()

    return shap_values
