# -*- coding: utf-8 -*-
"""本机 CPU 上 Metattack 算力瓶颈实测 + 外推。
Metattack 每步 = 在稠密邻接矩阵 A(N×N) 上跑 2 层 GCN forward/backward，
总迭代次数 ≈ n_perturbations × internal_epochs。
本脚本直接测"单次稠密 GCN 迭代"的真实耗时，再外推到目标数据集规模。
"""
import time, torch, torch.nn as nn, numpy as np, json, sys

print("torch threads:", torch.get_num_threads())

class DenseGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super().__init__()
        self.w1 = nn.Linear(nfeat, nhid, bias=False)
        self.w2 = nn.Linear(nhid, nclass, bias=False)
    def forward(self, A, X):
        H = torch.relu(A @ self.w1(X))
        Z = A @ self.w2(H)
        return Z

def bench(N, F=32, nhid=64, nclass=2, iters=15, device="cpu"):
    torch.manual_seed(0)
    A = torch.rand(N, N, dtype=torch.float32, device=device)
    X = torch.rand(N, F, dtype=torch.float32, device=device)
    y = torch.randint(0, nclass, (N,), device=device)
    m = DenseGCN(F, nhid, nclass).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    Z = m(A, X); (nn.CrossEntropyLoss()(Z, y)).backward(); opt.step()
    t0 = time.time()
    for _ in range(iters):
        Z = m(A, X)
        loss = nn.CrossEntropyLoss()(Z, y)
        opt.zero_grad(); loss.backward(); opt.step()
    dt = (time.time() - t0) / iters
    A_gb = A.numel() * 4 / 1e9
    del A, X, m, opt
    return dt, A_gb

Ns = [1000, 2000, 4000, 8000, 16000]
rows = []
print(f"{'N':>8} {'denseA_GB':>10} {'ms/iter':>10}")
for N in Ns:
    dt, agb = bench(N)
    rows.append({"N": N, "denseA_GB": round(agb,3), "sec_per_iter": round(dt,4)})
    print(f"{N:>8} {agb:>10.3f} {dt*1000:>10.1f}")
    sys.stdout.flush()

lns = np.log([r["N"] for r in rows]); lts = np.log([r["sec_per_iter"] for r in rows])
p, logc = np.polyfit(lns, lts, 1); c = np.exp(logc)
print(f"\n拟合: sec/iter = {c:.3e} * N^{p:.2f}")

# 真实规模（来自 data/raw/Fraud/*/processed/data_heteroFalse.pt）
targets = {
    "YelpChi": dict(N=45954, E=7693958, F=32),
    "Amazon":  dict(N=11944, E=8796784, F=25),
}
ep = 100  # Metattack 默认 internal epochs
print(f"\n--- 外推（internal_epochs={ep}）---")
for name, t in targets.items():
    N, E = t["N"], t["E"]
    tpi = c * (N ** p)                      # 单次稠密 GCN 迭代秒数
    dense_gb = (N*N*4)/1e9
    # 场景A：5% 边预算（原 run_attack.py 口径）—— 对 Metattack 极不合理但列出
    nA = int(0.05 * E)
    # 场景B：2% 节点预算（实际可跑的代表性预算）
    nB = int(0.02 * N)
    totA = tpi * nA * ep / 3600
    totB = tpi * nB * ep / 3600
    # 安全系数：Metattack 元梯度还有额外开销，乘 3
    print(f"\n{name}: N={N}, E={E}, 稠密A 内存={dense_gb:.1f} GB, 实测推算 ms/iter≈{tpi*1000:.1f}")
    print(f"   [5%边预算] n_pert={nA} -> 约 {totA:.0f} 小时 (×3安全={totA*3:.0f}h)")
    print(f"   [2%节点预算] n_pert={nB} -> 约 {totB:.1f} 小时 (×3安全={totB*3:.1f}h)")
    rows.append(dict(extrap=name, N=N, denseA_GB=round(dense_gb,1),
                     sec_per_iter_est=round(tpi,4), n_pert_5pct=nA, hours_5pct_x3=round(totA*3,0),
                     n_pert_2pctN=nB, hours_2pctN_x3=round(totB*3,1)))

json.dump(rows, open("src/common/attacks/_bench_metattack.json","w"), indent=2)
print("\nwritten src/common/attacks/_bench_metattack.json")
