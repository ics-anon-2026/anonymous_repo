# -*- coding: utf-8 -*-
"""方向二：训练与评估工具（处理类别不平衡）。"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _model_forward(model, data):
    """根据模型类型选择输入边格式。"""
    from gnn import HAN, BinaryGAT, BinaryGCN
    if isinstance(model, HAN):
        return model(data.x, data.edge_index_dict)
    return model(data.x, data.edge_index)


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, device="cpu"):
    """Focal loss for binary (2-class) classification."""
    ce = F.cross_entropy(inputs, targets, reduction="none")
    pt = torch.exp(-ce)
    alpha_t = torch.where(targets == 1,
                          torch.tensor(alpha, device=device),
                          torch.tensor(1 - alpha, device=device))
    loss = alpha_t * (1 - pt) ** gamma * ce
    return loss.mean()


def train_epoch(model, data, optimizer, device="cpu", pos_weight=None,
                use_focal=False, balanced_batch=False):
    model.train()
    data = data.to(device)
    out = _model_forward(model, data)
    is_binary_logit = out.dim() == 1

    if balanced_batch:
        # Sample equal numbers of fraud and normal nodes from train set
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        y_train = data.y[train_idx]
        pos_idx = train_idx[y_train == 1]
        neg_idx = train_idx[y_train == 0]
        n_pos = pos_idx.size(0)
        n_neg = neg_idx.size(0)
        n_sample = min(n_pos, n_neg)
        if n_sample > 0:
            perm_pos = torch.randperm(n_pos)[:n_sample]
            perm_neg = torch.randperm(n_neg)[:n_sample]
            sample_idx = torch.cat([pos_idx[perm_pos], neg_idx[perm_neg]])
        else:
            sample_idx = train_idx
        target = data.y[sample_idx]
    else:
        sample_idx = data.train_mask
        target = data.y[sample_idx]

    if is_binary_logit:
        # BCEWithLogitsLoss for binary models
        target_f = target.float()
        pw = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(out[sample_idx], target_f,
                                                   pos_weight=pw, reduction="mean")
    elif use_focal:
        loss = focal_loss(out[sample_idx], target, device=device)
    elif pos_weight is not None:
        weight = torch.tensor([1.0, pos_weight], device=device)
        loss = F.cross_entropy(out[sample_idx], target, weight=weight)
    else:
        loss = F.cross_entropy(out[sample_idx], target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach())


def _best_threshold_f1(y_true, y_prob, min_thr=0.05, max_thr=0.95):
    """根据验证集概率选择使 F1 最大的阈值，并限制在合理范围内。"""
    from sklearn.metrics import f1_score
    # Grid search over clipped thresholds for robustness
    thresholds = np.linspace(min_thr, max_thr, 100)
    best_f1, best_thr = 0.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return best_thr


@torch.no_grad()
def evaluate(model, data, mask, device="cpu", threshold=None):
    model.eval()
    data = data.to(device)
    out = _model_forward(model, data)
    y_true = data.y[mask].cpu().numpy()

    if out.dim() == 1:
        # Binary logit output
        y_prob = torch.sigmoid(out[mask]).cpu().numpy()
        if threshold is None:
            threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)
    else:
        probs = F.softmax(out, dim=1)[:, 1]
        y_prob = probs[mask].cpu().numpy()
        if threshold is not None:
            y_pred = (y_prob >= threshold).astype(int)
        else:
            pred = out.argmax(dim=1)
            y_pred = pred[mask].cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0
    return acc, f1, auc


@torch.no_grad()
def evaluate_with_val_threshold(model, data, test_mask, device="cpu"):
    """在验证集上选最佳阈值，然后在测试集上评估。"""
    model.eval()
    data = data.to(device)
    out = _model_forward(model, data)
    if out.dim() == 1:
        probs = torch.sigmoid(out)
    else:
        probs = F.softmax(out, dim=1)[:, 1]

    y_val = data.y[data.val_mask].cpu().numpy()
    p_val = probs[data.val_mask].cpu().numpy()
    threshold = _best_threshold_f1(y_val, p_val)

    return evaluate(model, data, test_mask, device, threshold=threshold), threshold
