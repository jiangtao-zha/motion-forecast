import torch
import torch.nn.functional as F


def WinTakeAllLoss(out, data):
    # y_hat [B num_q T 2]
    # pi 没有经过softmax处理 [B 1]
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
    loss += agent_reg_loss + agent_cls_loss

    others_reg_mask = ~data["x_padding_mask"][:, 1:, 50:]
    others_reg_loss = F.smooth_l1_loss(
        y_hat_others[others_reg_mask], y_others[others_reg_mask])
    loss += others_reg_loss

    return loss, agent_reg_loss, agent_cls_loss, others_reg_loss
