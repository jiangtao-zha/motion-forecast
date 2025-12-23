import torch
import torch.nn.functional as F


def WinTakeAllLoss(out, data):
    # y_hat [B num_q T 2]
    # pi 没有经过softmax处理 [B C]
    # y_hat_others [B N-1 T 2]
    y_hat, pi, y_hat_others = out["y_hat"], out["pi"].squeeze(
        2), out["y_hat_others"]
    y, y_others = data["y_diff"][:, 0], data["y_diff"][:, 1:]

    loss = 0
    B = y_hat.shape[0]
    B_range = range(B)

    l2_norm = torch.norm(y.unsqueeze(1) - y_hat, dim=-1).sum(-1)  # [B num_q]
    best_mode = torch.argmin(l2_norm, dim=-1)  # [B]

    y_hat_best = y_hat[B_range, best_mode]  # [B T 2]
    agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
    agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

    loss = agent_reg_loss + agent_cls_loss

    # l2_norm_mean = torch.norm(y.unsqueeze(1) - y_hat, dim=-1).mean(-1)  # [B num_q]
    # agent_cls_loss = F.smooth_l1_loss(pi, l2_norm_mean.detach(), reduction='mean')
    # loss += agent_reg_loss + agent_cls_loss

    # l2_norm_mean = torch.norm(y.unsqueeze(1) - y_hat, dim=-1).mean(-1)  # [B num_q]
    # scores = -l2_norm_mean.detach()
    # T = 0.1
    # soft_target = F.softmax(scores / T, dim=-1)
    # log_pi = F.log_softmax(pi, dim=-1)
    # agent_cls_loss = F.kl_div(log_pi, soft_target, reduction='batchmean')
    # loss += agent_reg_loss + agent_cls_loss

    others_reg_mask = ~data["x_padding_mask"][:, 1:, 50:]
    others_reg_loss = F.smooth_l1_loss(
        y_hat_others[others_reg_mask], y_others[others_reg_mask])
    loss += others_reg_loss

    return loss, agent_reg_loss, agent_cls_loss, others_reg_loss

def WinTakeAllLoss_Laplace(out, data):
    # ---------------------------------------------------------------------
    # out["y_hat"] 的形状必须是 [B, num_q, T, 4]
    # 最后一维 4 分别代表: [mu_x, mu_y, log_sigma_x, log_sigma_y]
    # ---------------------------------------------------------------------
    
    # y_hat: [B, num_q, T, 4]
    y_hat, pi, y_hat_others = out["y_hat"], out["pi"].squeeze(2), out["y_hat_others"]
    y, y_others = data["y_diff"][:, 0], data["y_diff"][:, 1:]

    loss = 0
    B = y_hat.shape[0]
    B_range = range(B)

    
    # mu: [B, num_q, T, 2], log_scale: [B, num_q, T, 2]
    mu = y_hat[..., :2]
    log_scale = y_hat[..., 2:]
    
    # 限制 log_scale 的范围以保证数值稳定性 (防止 b 过小导致除以0，或过大导致梯度消失)
    log_scale = torch.clamp(log_scale, min=-9.0, max=5.0)
    scale = torch.exp(log_scale) # b = exp(log_b)

    # 注意：通常依然使用"欧氏距离"来决定谁是 Winner，而不是用 NLL，这样更稳定
    l2_norm = torch.norm(y.unsqueeze(1) - mu, dim=-1).sum(-1)  # [B num_q]
    best_mode = torch.argmin(l2_norm, dim=-1)  # [B]

    # 提取 Winner 模态的 mu 和 scale
    # [B, T, 2]
    mu_best = mu[B_range, best_mode] 
    scale_best = scale[B_range, best_mode]
    log_scale_best = log_scale[B_range, best_mode]

    # 计算 Laplace NLL Loss (Agent Regression)
    # Laplace NLL 公式: log(2b) + |y - mu| / b
    # 省略常数 log(2)，Loss = log(b) + |y - mu| / b
    dist = torch.abs(y - mu_best) # L1 distance [B, T, 2]
    nll_loss = log_scale_best + dist / scale_best 
    
    # 对时间 T 和 坐标维度 2 求和/平均
    agent_reg_loss = nll_loss.mean() # 或者 .sum()，取决于你的 loss 权重配置

    # 分类 Loss (Classification)
    agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

    loss = agent_reg_loss + agent_cls_loss * 1.0 # 这里的权重 lambda 可能需要根据 NLL 的数值量级调整

    # 6其他 Agent 的 Loss (Others Regression)
    others_reg_mask = ~data["x_padding_mask"][:, 1:, 50:]
    others_reg_loss = F.smooth_l1_loss(
        y_hat_others[others_reg_mask], y_others[others_reg_mask])
    
    loss += others_reg_loss

    return loss, agent_reg_loss, agent_cls_loss, others_reg_loss