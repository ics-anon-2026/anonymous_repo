# -*- coding: utf-8 -*-
"""受害模型训练 CPU 微基准：真实 YelpChi/Amazon 稀疏图上 GCN/GAT 每轮耗时。
目的：确认攻击矩阵的另一半成本（victim 训练）在本机 CPU 可行。
"""
import time, torch, torch.nn as nn, torch.nn.functional as F, glob
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import torch_geometric.transforms as T

def load(name):
    p = glob.glob(f"data/**/{name}/processed/data_heteroFalse.pt", recursive=True)[0]
    d = torch.load(p, weights_only=False)[0]
    return d

class GCN(nn.Module):
    def __init__(self, F, nhid=64, nclass=2):
        super().__init__()
        self.c1 = GCNConv(F, nhid); self.c2 = GCNConv(nhid, nclass)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei)); x = F.dropout(x, 0.5, self.training)
        return self.c2(x, ei)

class GAT(nn.Module):
    def __init__(self, F, nhid=8, nclass=2, heads=8):
        super().__init__()
        self.c1 = GATConv(F, nhid, heads=heads); self.c2 = GATConv(nhid*heads, nclass, heads=1)
    def forward(self, x, ei):
        x = F.elu(self.c1(x, ei)); x = F.dropout(x, 0.6, self.training)
        return self.c2(x, ei)

def bench_model(name, data, cls, epochs=8):
    torch.manual_seed(0)
    m = cls(data.num_features)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    tr = data.train_mask; vl = getattr(data, 'val_mask', None); te = getattr(data, 'test_mask', None)
    y = data.y
    # warmup 2
    for _ in range(2):
        m.train(); opt.zero_grad(); out = m(data.x, data.edge_index)
        loss = F.cross_entropy(out[tr], y[tr]); loss.backward(); opt.step()
    t0 = time.time()
    for _ in range(epochs):
        m.train(); opt.zero_grad(); out = m(data.x, data.edge_index)
        loss = F.cross_entropy(out[tr], y[tr]); loss.backward(); opt.step()
    dt = (time.time()-t0)/epochs
    return dt

print("torch threads:", torch.get_num_threads())
for name in ["YelpChi", "Amazon"]:
    d = load(name)
    print(f"\n{name}: {d.num_nodes} nodes, {d.num_edges} edges")
    for label, cls in [("GCN", GCN), ("GAT", GAT)]:
        dt = bench_model(name, d, cls)
        est200 = dt*200/60
        print(f"  {label}: {dt*1000:.0f} ms/epoch -> 200 epoch ≈ {est200:.1f} 分钟/次victim训练")
