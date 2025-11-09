import torch
import torch.nn as nn
from model.Encoder import SeptEncoder
from model.Decoder import SeptDecoder


class SEPT(nn.Module):
    def __init__(self, agent_input_dim: int,
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
                 activation='gelu'):
        super().__init__()
        self.encoder = SeptEncoder(agent_dim=agent_input_dim,
                                   road_dim=road_input_dim,
                                   num_layers_Kt=num_layers_Kt,
                                   num_layers_Ks=num_layers_Ks,
                                   d_model=d_model,
                                   num_head_Kt=num_head_Kt,
                                   num_head_Ks=num_head_Ks,
                                   dim_feedforward=dim_feedforward,
                                   dropout=dropout,
                                   activation=activation)

        self.decoder = SeptDecoder(num_queries=num_queries,
                                   num_layers_Kc=num_layers_Kc,
                                   d_model=d_model,
                                   num_head_Kc=num_head_Kc,
                                   dim_feedforward=dim_feedforward,
                                   dropout=dropout,
                                   activation=activation)

    def forward(self, src_agent, src_road, agent_pos_feat, road_pos_feat, agent_key_padding_mask=None, agent_padding_mask=None, road_key_padding_mask=None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        encode_x = self.encoder(src_agent=src_agent,
                                src_road=src_road,
                                agent_pos_feat=agent_pos_feat,
                                road_pos_feat=road_pos_feat,
                                agent_key_padding_mask=agent_key_padding_mask,
                                agent_padding_mask=agent_padding_mask,
                                road_key_padding_mask=road_key_padding_mask
                                )
        # assert not torch.isnan(encode_x).any(), "encode_x has nan"
        memory_key_padding_mask = None
        if agent_key_padding_mask is not None and road_key_padding_mask is not None:
            memory_key_padding_mask = torch.cat(
                [agent_key_padding_mask, road_key_padding_mask], dim=1)
        trajectory, probability = self.decoder(memory=encode_x,
                                               memory_mask=None,
                                               memory_key_padding_mask=memory_key_padding_mask)
        # assert not torch.isnan(trajectory).any(), "trajectory has nan"

        return trajectory, probability


if __name__ == '__main__':
    from Encoder import SeptEncoder
    from Decoder import SeptDecoder
    print("--- 开始测试 SEPT 模型 ---")

    # 1. 定义超参数 (与 Encoder 测试保持一致，并添加 Decoder 参数)
    BATCH_SIZE = 4
    SEQ_A = 10     # Max agents
    SEQ_R = 50     # Max road segments
    TIME = 50      # History length
    AGENT_DIM = 8
    ROAD_DIM = 6
    D_MODEL = 128
    NUM_LAYERS_KT = 2
    NUM_LAYERS_KS = 2
    NUM_LAYERS_KC = 2  # Decoder layers
    KT_NUM_HEAD = 8
    KS_NUM_HEAD = 8
    num_head_Kc = 8   # Decoder heads
    NUM_QUERIES = 6  # Decoder queries
    DIM_FEEDFORWARD = 256
    DROPOUT = 0.1
    PRED_LEN = 60  # 假设 Decoder 输出 60 步轨迹

    # 2. 实例化模型
    try:
        model = SEPT(
            agent_input_dim=AGENT_DIM,
            road_input_dim=ROAD_DIM,
            num_layers_Kt=NUM_LAYERS_KT,
            num_layers_Ks=NUM_LAYERS_KS,
            num_layers_Kc=NUM_LAYERS_KC,
            d_model=D_MODEL,
            num_head_Kt=KT_NUM_HEAD,  # 传递给 Encoder
            num_head_Ks=KS_NUM_HEAD,  # 传递给 Encoder
            num_head_Kc=num_head_Kc,   # 传递给 Decoder
            num_queries=NUM_QUERIES,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT        # 统一使用 dropout
        )
        model.eval()  # 设置为评估模式，会禁用 Dropout
        print("模型实例化成功。")
    except Exception as e:
        print(f"模型实例化失败: {e}")
        exit()  # 如果实例化失败，则退出

    # 3. 创建虚拟输入数据 (与 Encoder 测试相同)
    src_agent = torch.randn(BATCH_SIZE, SEQ_A, TIME, AGENT_DIM)
    src_road = torch.randn(BATCH_SIZE, SEQ_R, ROAD_DIM)
    agent_padding_mask = torch.zeros(BATCH_SIZE, SEQ_A, TIME, dtype=torch.bool)
    agent_padding_mask[0, 1, 30:] = True
    agent_padding_mask[1, 0, 40:] = True
    agent_key_padding_mask = torch.zeros(BATCH_SIZE, SEQ_A, dtype=torch.bool)
    agent_key_padding_mask[0, 8:] = True
    agent_key_padding_mask[1, 5:] = True
    road_key_padding_mask = torch.zeros(BATCH_SIZE, SEQ_R, dtype=torch.bool)
    road_key_padding_mask[0, 40:] = True
    road_key_padding_mask[1, 30:] = True

    print("\n--- 打印输入形状 ---")
    print(f"src_agent 形状: \t\t{src_agent.shape}")
    print(f"src_road 形状: \t\t{src_road.shape}")
    print(f"agent_padding_mask 形状: \t{agent_padding_mask.shape}")
    print(f"agent_key_padding_mask 形状: \t{agent_key_padding_mask.shape}")
    print(f"road_key_padding_mask 形状: \t{road_key_padding_mask.shape}")

    # 4. 执行前向传播 (在 no_grad 上下文中，因为只是测试形状)
    try:
        with torch.no_grad():
            trajectory, probability = model(
                src_agent=src_agent,  # 确保参数名匹配 forward 定义
                src_road=src_road,
                agent_key_padding_mask=agent_key_padding_mask,
                agent_padding_mask=agent_padding_mask,
                road_key_padding_mask=road_key_padding_mask
            )

        print("\n--- 前向传播成功 ---")
        print(f"输出 trajectory 形状: \t{trajectory.shape}")
        print(f"输出 probability 形状: \t{probability.shape}")

        # 5. 验证输出形状
        expected_traj_shape = (BATCH_SIZE, NUM_QUERIES, PRED_LEN, 2)
        # 概率形状取决于 Decoder 的 MLP 输出是 [B, N, 1] 还是 [B, N]
        expected_prob_shape1 = (BATCH_SIZE, NUM_QUERIES, 1)
        expected_prob_shape2 = (BATCH_SIZE, NUM_QUERIES)

        assert trajectory.shape == expected_traj_shape, f"轨迹形状不匹配! 预期 {expected_traj_shape}, 得到 {trajectory.shape}"
        assert probability.shape == expected_prob_shape1 or probability.shape == expected_prob_shape2, \
            f"概率形状不匹配! 预期 {expected_prob_shape1} 或 {expected_prob_shape2}, 得到 {probability.shape}"

        print(f"输出形状验证成功!")

    except Exception as e:
        print(f"\n--- 前向传播失败 ---")
        import traceback  # 打印更详细的错误栈
        traceback.print_exc()
        # print(f"错误信息: {e}")
