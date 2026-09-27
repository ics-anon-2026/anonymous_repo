# -*- coding: utf-8 -*-
"""方向二：异构图金融欺诈检测基线模型。"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HANConv


class GCN(torch.nn.Module):
    """同质图 GCN 基线（把异构图当作同构图处理）。"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x


class GAT(torch.nn.Module):
    """GAT 基线。"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class HAN(torch.nn.Module):
    """HAN 基线，需要输入元路径 edge_index_dict。"""
    def __init__(self, in_channels, hidden_channels, out_channels, metadata=None,
                 heads=8, num_layers=2, dropout=0.6):
        super().__init__()
        self.han = HANConv(in_channels, hidden_channels, heads=heads, dropout=dropout,
                           metadata=metadata)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None, edge_weight=None):
        # 支持两种调用：传入 tensor x（同质图）或 x_dict
        if isinstance(x, dict):
            x_dict = x
            edge_index_dict = edge_index
        else:
            x_dict = {"user": x}
            edge_index_dict = edge_index
        out_dict = self.han(x_dict, edge_index_dict)
        # 假设只有一个节点类型
        x_out = list(out_dict.values())[0]
        return self.classifier(x_out)


class BinaryGAT(torch.nn.Module):
    """GAT with single logit output for BCE stability."""
    def __init__(self, in_channels, hidden_channels, heads=8, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, 1, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x.squeeze(-1)


class BinaryGCN(torch.nn.Module):
    """GCN with single logit output for BCE stability."""
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return x.squeeze(-1)
