from model.SEPT_insert import SEPT
import pytorch_lightning as L
from model.Loss import WinTakeAllLoss
from metrics import minADE, minFDE, brierMinFDE, MR
import torch.optim as optim
from torchmetrics import MetricCollection
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
# torch.set_printoptions(profile="full", threshold=100000, linewidth=100)


class SEPT_Module(L.LightningModule):
    def __init__(self,
                 agent_input_dim: int,
                 road_input_dim: int,
                 num_layers_Kt: int,
                 num_layers_Ks: int,
                 num_layers_Kc: int,
                 d_model: int,
                 num_head_Kt: int,
                 num_head_Ks: int,
                 num_head_Kc: int,
                 num_queries: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 activation='gelu',
                 learning_rate=1e-4,
                 weight_decay=0.01,
                 train_batch_size: int = 32,
                 warmup_steps: int = 1000,
                 start_lr_ratio:int = 0,
                 min_learning_rate:int = 0):
        super().__init__()
        self.model = SEPT(agent_input_dim=agent_input_dim,
                          road_input_dim=road_input_dim,
                          num_layers_Kt=num_layers_Kt,
                          num_layers_Ks=num_layers_Ks,
                          num_layers_Kc=num_layers_Kc,
                          d_model=d_model,
                          num_head_Kt=num_head_Kt,
                          num_head_Ks=num_head_Ks,
                          num_head_Kc=num_head_Kc,
                          num_queries=num_queries,
                          dim_feedforward=dim_feedforward,
                          dropout=dropout,
                          activation=activation)
        self.model.apply(_init_weights_recursive)

        k_value = num_queries
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        metrics = MetricCollection({
            f'minADE{k_value}': minADE(k=k_value),
            f'minFDE{k_value}': minFDE(k=k_value),
            # f'minADE{1}': minADE(k=1),
            # f'minFDE{1}': minFDE(k=1),
            f'brierMinFDE{k_value}': brierMinFDE(k=k_value),
            'MR': MR()
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        out = self.model(batch)

        loss, loss_reg, loss_cls, loss_other = WinTakeAllLoss(out, batch)

        # 准备 metric 输入
        # 需要将 probability (logits) 转换为概率
        prob_softmax = torch.softmax(
            out["pi"].squeeze(-1), dim=-1)  # 形状 [B, N]
        outputs_for_metrics = {"y_hat": out["y_hat"], "pi": prob_softmax}
        target_for_metrics = batch['y_diff'][:, 0, :, :]

        # 更新 metric 状态
        self.train_metrics.update(outputs_for_metrics, target_for_metrics)

        # 记录损失
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_reg_loss", loss_reg, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log("train_cls_loss", loss_cls, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log("train_other_loss", loss_other, on_step=True,
                 on_epoch=True, sync_dist=True)

        # 记录 metrics (在 epoch 结束时计算并记录)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch)

        loss, loss_reg, loss_cls, loss_other = WinTakeAllLoss(out, batch)

        # 准备 metric 输入
        # 需要将 probability (logits) 转换为概率
        prob_softmax = torch.softmax(
            out["pi"].squeeze(-1), dim=-1)  # 形状 [B, N]
        outputs_for_metrics = {"y_hat": out["y_hat"], "pi": prob_softmax}
        target_for_metrics = batch['y_diff'][:, 0, :, :]

        # 更新验证 metric
        self.val_metrics.update(outputs_for_metrics, target_for_metrics)

        # 记录验证损失 (修正键名)
        self.log("val_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_reg_loss", loss_reg, on_step=False,
                 on_epoch=True, sync_dist=True)
        self.log("val_cls_loss", loss_cls,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_other_loss", loss_other, on_step=True,
                 on_epoch=True, sync_dist=True)

        # 记录验证 metrics (在 epoch 结束时计算并记录)
        self.log_dict(self.val_metrics, on_step=False,
                      on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        target_gt = batch['y_diff'][:, 0, :, :]
        out = self.model(batch)

        # 2. 准备 Metric 输入
        prob_softmax = torch.softmax(out["pi"].squeeze(-1), dim=-1)
        outputs_for_metrics = {"y_hat": out["y_hat"], "pi": prob_softmax}

        # 3. 更新测试集指标状态
        self.test_metrics.update(outputs_for_metrics, target_gt)

        # 4. 记录测试集指标 (通常只记录在 epoch 结束时)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def linear_warmup_lambda(self, step: int):
        """
        计算 Warmup 阶段的学习率比例。
        在 step=0 时，比例为 start_ratio。
        在 step=warmup_steps 时，比例为 1.0。
        """
        start_ratio = 0.1  # 假设你想让起始学习率是 max_lr 的 10%
        if step < self.hparams.warmup_steps:
            # 计算当前步数的增量比例： (1.0 - start_ratio) * (step / warmup_steps)
            # 总比例 = start_ratio + 增量比例
            return start_ratio + (1.0 - start_ratio) * (step / max(1.0, self.hparams.warmup_steps))
        else:
            # 在 warmup 结束时，比例因子锁定在 1.0
            return 1.0

    def configure_optimizers(self):
        # 使用 self.hparams 访问保存的超参数 (如果使用了 save_hyperparameters)
        # optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # 或者直接指定
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.trainer.datamodule:
            total_steps = self.trainer.estimated_stepping_batches

        # 调度器 1: 线性预热
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=self.linear_warmup_lambda
        )

        # 调度器 2: 余弦衰减 (从 warmup 结束时开始)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(total_steps - self.hparams.warmup_steps),
            eta_min=1e-6
        )

        # 3. 使用 SequentialLR 组合它们
        # (需要 torch 1.7+，您的环境肯定满足)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_steps]  # 在 warmup_steps 步数时切换调度器
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 关键：指定调度器按 "step" 更新
                "frequency": 1,      # 关键：每个 step 都更新
            }
        }


def _init_weights_recursive(m: nn.Module):
    # 1. 处理线性层 (Linear Layers)
    if isinstance(m, nn.Linear):
        # Kaiming/He 初始化通常适用于 ReLU 或其变体 (如 GELU)
        # 对于 Transformer，有时也用 Xavier/Glorot 初始化
        nn.init.xavier_uniform_(m.weight)
        # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # 初始化偏置为 0
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # 2. 处理嵌入层 (Embedding Layers)
    elif isinstance(m, nn.Embedding):
        # 嵌入层常用较小的正态分布初始化
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # 如果有 padding_idx，可以将其初始化为 0
        # if m.padding_idx is not None:
        #     with torch.no_grad():
        #         m.weight[m.padding_idx].fill_(0)

    # 3. 处理层归一化 (LayerNorm)
    elif isinstance(m, nn.LayerNorm):
        # 归一化层的 scale (weight) 初始化为 1, shift (bias) 初始化为 0
        if m.weight is not None:  # elementwise_affine=True (默认)
            nn.init.ones_(m.weight)
        if m.bias is not None:   # elementwise_affine=True (默认)
            nn.init.zeros_(m.bias)


def nan_hook(module, input, output):
    # output 可能是 tuple (如 LSTM)，也可能是 Tensor
    outputs_to_check = []
    if isinstance(output, torch.Tensor):
        outputs_to_check.append(output)
    elif isinstance(output, tuple):
        outputs_to_check.extend(
            t for t in output if isinstance(t, torch.Tensor))

    for i, out in enumerate(outputs_to_check):
        if torch.isnan(out).any():
            print(
                f"!!! NaN found after module: {module.__class__.__name__} (Output {i}) !!!")
            # 可以在这里设置断点或抛出异常
            # import pdb; pdb.set_trace()
            raise RuntimeError(f"NaN detected in {module}")
