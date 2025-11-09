import torch
import torch.nn.functional as F


def WinTakeAllLoss(trajectory, probability_logits, y_hat, weight_reg: float = 1.0,
                   weight_cls: float = 1.0,label_smoothing:float = 0.1):
    """
    计算 "赢者通吃" 损失，结合了最佳轨迹的回归损失和分类损失。

    Args:
        trajectory (torch.Tensor): 预测的 N 条轨迹, 形状 (B, N, T, 2)。
        probability_logits (torch.Tensor): N 条轨迹的预测得分 (Logits), 形状(B, N, 1)。
        y_hat (torch.Tensor): 真实轨迹, 形状 (B, T, 2)。
        weight_reg (float): 回归损失的权重。
        weight_cls (float): 分类损失的权重。

    Returns:
        tuple: (total_loss, loss_reg, loss_cls)
    """
    B, N, T, _ = trajectory.shape

    assert trajectory.size(1) == probability_logits.size(1), \
        f"轨迹模式数 ({trajectory.size(1)}) 与概率模式数 ({probability_logits.size(1)}) 不匹配"

    assert trajectory.size(2) == y_hat.size(1), \
        f"预测时间步长 ({trajectory.size(2)}) 与目标时间步长 ({y_hat.size(1)}) 不匹配"
    assert y_hat.size(0) == B


    ade = torch.norm(
        trajectory[..., :2] - y_hat.unsqueeze(1)[..., :2], p=2, dim=-1
    ).mean(-1)  # [B N]

    min_index = ade.argmin(dim=-1)  # [B]

    idx_exp = min_index.view(B, 1, 1, 1).expand(-1, -1, T, 2)
    best_trajectory = torch.gather(
        trajectory, dim=1, index=idx_exp).squeeze(1)  # [B, T, 2]
    
    loss_reg = F.l1_loss(
        best_trajectory, y_hat, reduction='mean') * weight_reg
    # assert not torch.isnan(loss_reg).any(), "loss_reg has nan"

    max_logit = probability_logits.max()
    loss_cls = F.cross_entropy(
        probability_logits.squeeze(-1), min_index, reduction='mean',label_smoothing = label_smoothing) * weight_cls
    # assert not torch.isnan(loss_cls).any(), "loss_cls has nan"
    loss = loss_reg + loss_cls

    return loss, loss_reg, loss_cls


if __name__ == '__main__':
    B, N, T = 4, 6, 60  # Batch=4, 预测6种模式, 预测60步

    # 模拟输入
    pred_traj = torch.randn(B, N, T, 2)
    pred_prob_logits = torch.randn(B, N)  # Logits
    gt_traj = torch.randn(B, T, 2)

    # 计算损失
    total_loss, reg_loss, cls_loss = WinTakeAllLoss(
        pred_traj, pred_prob_logits, gt_traj)

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Regression Loss: {reg_loss.item():.4f}")
    print(f"Classification Loss: {cls_loss.item():.4f}")
