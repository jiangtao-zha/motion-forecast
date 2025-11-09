import torch
import torch.nn as nn


class CrossAttender(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_head: int, dim_feedforward: int, dropout: float = 0.1, activation='gelu'):
        super().__init__()
        self.decoderlayer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(
            self.decoderlayer, num_layers=num_layers)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        参数:
            tgt (Tensor): 目标序列 (在这里是可学习查询)，形状 (N, T, E)。
            memory (Tensor): 编码器的输出序列 (场景记忆)，形状 (N, S, E)。
            tgt_mask (Tensor, optional): 目标序列的注意力掩码，形状 (T, T)。
            memory_mask (Tensor, optional): 记忆序列的注意力掩码，形状 (T, S)。
            tgt_key_padding_mask (Tensor, optional): 目标序列的填充掩码，形状 (N, T)。
            memory_key_padding_mask (Tensor, optional): 记忆序列的填充掩码，形状 (N, S)。
        返回:
            Tensor: 解码器的输出，形状 (N, T, E)。
        """
        x = self.decoder(tgt=tgt,
                         memory=memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        return x


class OutTrajectoryAndProbability(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, PreTime: int = 60, dropout: float = 0.1):
        super().__init__()
        self.mlp_trajectory = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(dim_feedforward, 2 * PreTime))

        self.mlp_probability = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                             nn.ReLU(),
                                             nn.Dropout(dropout),
                                             nn.Linear(dim_feedforward, 1))
        self.pre_time = PreTime

    def forward(self, src):
        trajectory = self.mlp_trajectory(src)
        B, A, _ = trajectory.shape
        trajectory = trajectory.view(B, A, self.pre_time, 2)

        probability = self.mlp_probability(src)
        return trajectory, probability


class SeptDecoder(nn.Module):
    """
    实现了 SEPT 论文中的解码器部分，包含可学习查询和 CrossAttender。
    """

    def __init__(self, num_queries: int, num_layers_Kc: int, d_model: int, num_head_Kc: int, dim_feedforward: int, dropout: float, activation='gelu'):
        super().__init__()
        self.cross_attender = CrossAttender(num_layers=num_layers_Kc,
                                            d_model=d_model,
                                            n_head=num_head_Kc,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                            activation=activation)
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.mlp_decoder = OutTrajectoryAndProbability(d_model=d_model,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout)

    def forward(self,
                memory: torch.Tensor,
                memory_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        参数:
            memory (Tensor): 编码器的输出序列 (场景记忆)，形状 (N, S, E)。
            memory_mask (Tensor, optional): 记忆序列的注意力掩码，形状 (T, S)。
            memory_key_padding_mask (Tensor, optional): 记忆序列的填充掩码，形状 (N, S)。
        返回:
            Tensor: 解码器的输出，形状 (N, num_queries, E)。
        """
        batch_size = memory.size(0)
        batch_queries = self.queries.expand(batch_size, -1, -1)
        x = self.cross_attender(
            tgt=batch_queries,
            memory=memory,  
            tgt_mask=None,
            memory_mask=memory_mask,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=memory_key_padding_mask
        )
        # x = torch.nan_to_num(x, nan=0.0)
        # print(f"x size: {x.shape}")
        trajectory, probability = self.mlp_decoder(x)
        return trajectory, probability


if __name__ == '__main__':
    # 定义超参数
    NUM_QUERIES = 6     # 预测6条轨迹
    NUM_LAYERS_KC = 2   # 解码器层数
    D_MODEL = 256     # 特征维度
    num_head_Kc = 8       # 解码器头数
    DIM_FF = 512      # FFN中间层维度
    DROPOUT = 0.1     # 使用 float
    PRED_LEN = 60     # 预测时间步长

    BATCH_SIZE = 4
    NUM_SCENE_ELEMENTS = 120  # (A+S)，例如 20个智能体 + 100个车道段

    # 1. 实例化 SeptDecoder
    # [修正] 传递 float 类型的 dropout
    decoder = SeptDecoder(num_queries=NUM_QUERIES, num_layers_Kc=NUM_LAYERS_KC,
                          d_model=D_MODEL, num_head_Kc=num_head_Kc, 
                          dim_feedforward=DIM_FF, dropout=float(DROPOUT))

    print("SeptDecoder 实例化成功.")

    # 2. 创建虚拟的编码器输出 (memory) 和填充掩码
    dummy_memory = torch.randn(BATCH_SIZE, NUM_SCENE_ELEMENTS, D_MODEL)
    dummy_memory_key_padding_mask = torch.zeros(
        BATCH_SIZE, NUM_SCENE_ELEMENTS, dtype=torch.bool)
    dummy_memory_key_padding_mask[0, 100:] = True
    dummy_memory_key_padding_mask[1, 80:] = True
    dummy_memory_key_padding_mask[3, 90:] = True

    print(f"输入 memory 形状: {dummy_memory.shape}")
    print(
        f"输入 memory_key_padding_mask 形状: {dummy_memory_key_padding_mask.shape}")

    # 3. 执行前向传播
    # [修正] SeptDecoder 现在返回两个值
    trajectory_pred, probability_pred = decoder(
        memory=dummy_memory,
        memory_key_padding_mask=dummy_memory_key_padding_mask
    )

    print(f"\n--- 前向传播成功 ---")
    
    # [修正] 分别检查 trajectory 和 probability 的形状
    print(f"输出 trajectory 形状: {trajectory_pred.shape}")
    expected_traj_shape = (BATCH_SIZE, NUM_QUERIES, PRED_LEN, 2)
    print(f"(预期形状: {expected_traj_shape})")
    assert trajectory_pred.shape == expected_traj_shape

    print(f"输出 probability 形状: {probability_pred.shape}")
    expected_prob_shape = (BATCH_SIZE, NUM_QUERIES, 1)
    print(f"(预期形状: {expected_prob_shape})")
    assert probability_pred.shape == expected_prob_shape