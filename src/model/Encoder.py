
from pathlib import Path
from turtle import pos
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(profile="full", threshold=100000, linewidth=100)


class TempoNet(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_head: int, dim_feedforward: int, dropout: float, activation='gelu'):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

    def forward(self, src, mask=None, key_padding_mask=None) -> torch.Tensor:
        # src size : [batch seq d_model]
        # mask szie : 注意力掩码，一般在encoder里不用
        # key_padding_mask szie : [batch seq]
        encoder_out = self.encoder(src=src,
                                   mask=mask,
                                   src_key_padding_mask=key_padding_mask)

        return encoder_out


class SpaNet(nn.Module):
    def __init__(self, num_layers: int, d_model: int, n_head: int, dim_feedforward: int, dropout: float, activation='gelu'):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

    def forward(self, src, mask=None, key_padding_mask=None) -> torch.Tensor:
        # src size : [batch seq d_model]
        # mask szie : 注意力掩码，一般在encoder里不用
        # key_padding_mask szie : [batch seq]
        encoder_out = self.encoder(src, mask, key_padding_mask)

        return encoder_out


class SeptEncoder(nn.Module):
    def __init__(self, agent_dim: int, road_dim: int, num_layers_Kt: int, num_layers_Ks: int, d_model: int, num_head_Kt: int, num_head_Ks: int, dim_feedforward: int, dropout: float = 0.1, activation='gelu'):
        super().__init__()
        self.tempo_net = TempoNet(num_layers=num_layers_Kt,
                                  d_model=d_model,
                                  n_head=num_head_Kt,
                                  dim_feedforward=dim_feedforward,
                                  dropout=dropout,
                                  activation=activation)
        self.spa_net = SpaNet(num_layers=num_layers_Ks,
                              d_model=d_model,
                              n_head=num_head_Ks,
                              dim_feedforward=dim_feedforward,
                              dropout=dropout,
                              activation=activation)
        self.projection1 = nn.Linear(agent_dim, d_model)
        self.relu1 = nn.ReLU()
        self.norm_agent_proj = nn.LayerNorm(d_model)
        self.projection2 = nn.Linear(road_dim, d_model)
        self.relu2 = nn.ReLU()
        self.norm_road_proj = nn.LayerNorm(d_model)

        self.PositionEncoding = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, src_agent, src_road, agent_pos_feat,road_pos_feat, agent_key_padding_mask=None, agent_padding_mask=None, road_key_padding_mask=None) -> torch.Tensor:
        # src_road 形状: [batch, seq_r, road_dim]
        # src_agent 形状: [batch, seq_a, Time, agent_dim]
        # agent_pos_feat 包括agent_heading agent_center  Size:[batch,seq_r,4]
        # road_pos_feat road_heading  road_center Size:[batch,seq_l,4]
        # agent_key_padding_mask (maxpool 以及 spanet需要使用的agent级别的mask) : [batch, seq_a]
        # agent_padding_mask (智能体时间步的掩码): [batch, seq_a, Time]
        # road_key_padding_mask (道路的掩码): [batch, seq_r]

        # 在进行特征操作时，应该直接去除由于padding带来的空agent

        projected_agent = self.projection1(src_agent)
        # assert not torch.isnan(self.projection1.weight).any(), "self.projection1.weight has nan"
        x_agent_projection = self.relu1(projected_agent)
        x_agent_projection = self.norm_agent_proj(x_agent_projection)
        # x_agent_projection 形状: [batch, seq_a, Time, d_model]
        B, A, T, D = x_agent_projection.shape
        # x_agent_projection = x_agent_projection.view(B*A, T, -1)

        # 取反，True表示没有被mask的agent
        real_key_agent_mask = ~agent_key_padding_mask
        real_agent_feature = x_agent_projection[real_key_agent_mask]
        # real_agent_feature : [num_real_agent T D]

        # gather 关于real agent的时间掩码
        real_agent_time_mask = agent_padding_mask[real_key_agent_mask]
        # real_agent_time_mask : [num_real_agent T]

        x_agent_encode = self.tempo_net(
            real_agent_feature, key_padding_mask=real_agent_time_mask)

        x_agent_encode_full = torch.zeros(
            B, A, T, D, device=x_agent_projection.device, dtype=x_agent_projection.dtype)

        x_agent_encode_full[real_key_agent_mask] = x_agent_encode

        # x_agent_encode_full = torch.nan_to_num(x_agent_encode_full, nan=0.0)

        x_agent_maxpool = torch.max(x_agent_encode_full, dim=2).values
        # x_anget_maxpool : [batch seq_a d_model]

        # add position embedding 
        x_agent_maxpool = x_agent_maxpool + self.PositionEncoding(agent_pos_feat)

        # 排除由于batch拉大强行加入的pad_agent 这里面mask全都是 1
        # x_agent_maxpool[agent_key_padding_mask.unsqueeze(-1).expand(-1,-1,x_agent_maxpool.size(2))] = 0

        # x_anget_maxpool : [batch seq_a d_model]
        # road_process
        x_road_projection = self.relu2(self.projection2(src_road))
        x_road_projection = self.norm_road_proj(x_road_projection)
        # x_road_projection : [batch seq_r d_model]

        # add position embedding 
        x_road_projection = x_road_projection + self.PositionEncoding(road_pos_feat)

        # concat road and agent
        x = torch.concat([x_agent_maxpool, x_road_projection], dim=1)

        spa_padding_mask = None
        if agent_key_padding_mask is not None and road_key_padding_mask is not None:
            spa_padding_mask = torch.cat(
                [agent_key_padding_mask, road_key_padding_mask], dim=1)

        # SpaNet
        x = self.spa_net(x, key_padding_mask=spa_padding_mask)
        # x = torch.nan_to_num(x, nan=0.0)
        assert not torch.isnan(x).any(), "x has nan"
        return x


if __name__ == "__main__":
    print("--- 开始测试 SeptEncoder ---")

    # 1. 定义超参数
    BATCH_SIZE = 4
    SEQ_A = 10     # 批次中最多有 10 个智能体
    SEQ_R = 50     # 批次中最多有 50 个道路段
    TIME = 50      # 时间步长（历史）
    AGENT_DIM = 8  # 智能体原始特征维度
    ROAD_DIM = 6   # 道路原始特征维度
    D_MODEL = 128  # Transformer 内部维度

    # 2. 实例化模型
    model = SeptEncoder(
        agent_dim=AGENT_DIM,
        road_dim=ROAD_DIM,
        num_layers_Kt=2,
        num_layers_Ks=2,
        d_model=D_MODEL,
        num_head_Kt=8,
        num_head_Ks=8,
        dim_feedforward=256,
        dropout=0.1  # 使用 float
    )
    print("模型实例化成功。")

    # 3. 创建虚拟输入数据
    # 智能体数据 (B, A, T, Dim)
    src_agent = torch.zeros(BATCH_SIZE, SEQ_A, TIME, AGENT_DIM) * 100
    src_agent[..., 0] = 2
    # 道路数据 (B, R, Dim)
    src_road = torch.randn(BATCH_SIZE, SEQ_R, ROAD_DIM)

    # 4. 创建三个关键的掩码 (Masks)
    #    (False = 真实数据, True = 填充数据)

    # 4.1 agent_padding_mask (时间步级别) [B, A, T]
    # 模拟每个智能体的历史轨迹长度不同
    agent_padding_mask = torch.zeros(BATCH_SIZE, SEQ_A, TIME, dtype=torch.bool)
    # 举例：第0个批次的第1个智能体，只有30个时间步是有效的
    agent_padding_mask[0, 1, :] = True
    # 举例：第1个批次的第0个智能体，只有40个时间步是有效的
    agent_padding_mask[1, 0, 40:] = True

    # 4.2 agent_key_padding_mask (智能体级别) [B, A]
    # 模拟每个批次中的智能体数量不同
    agent_key_padding_mask = torch.zeros(BATCH_SIZE, SEQ_A, dtype=torch.bool)
    # 举例：第0个批次只有 8 个智能体
    agent_key_padding_mask[0, 8:] = True
    # 举例：第1个批次只有 5 个智能体
    agent_key_padding_mask[1, 5:] = True

    # 4.3 road_key_padding_mask (道路级别) [B, R]
    # 模拟每个批次中的道路段数量不同
    road_key_padding_mask = torch.zeros(BATCH_SIZE, SEQ_R, dtype=torch.bool)
    # 举例：第0个批次只有 40 个道路段
    road_key_padding_mask[0, 40:] = True
    # 举例：第1个批次只有 30 个道路段
    road_key_padding_mask[1, 30:] = True

    print("\n--- 打印输入形状 ---")
    print(f"src_agent 形状: \t\t{src_agent.shape}")
    print(f"src_road 形状: \t\t{src_road.shape}")
    print(f"agent_padding_mask 形状: \t{agent_padding_mask.shape}")
    print(f"agent_key_padding_mask 形状: \t{agent_key_padding_mask.shape}")
    print(f"road_key_padding_mask 形状: \t{road_key_padding_mask.shape}")

    # 5. 执行前向传播
    try:
        output = model(
            src_agent,
            src_road,
            agent_key_padding_mask,
            agent_padding_mask,
            road_key_padding_mask
        )
        assert not torch.isnan(output).any(), "output has NaN!"

        print("\n--- 前向传播成功 ---")
        print(f"输出张量形状: \t\t{output.shape}")

        # 6. 验证输出形状
        expected_shape = (BATCH_SIZE, SEQ_A + SEQ_R, D_MODEL)
        assert output.shape == expected_shape
        print(f"输出形状验证成功! 预期: {expected_shape}")

    except Exception as e:
        print(f"\n--- 前向传播失败 ---")
        print(f"错误信息: {e}")
