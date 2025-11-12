import torch
import torch.nn as nn
from model.lane_embedding import LaneEmbeddingLayer


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
        # Encoder

        # TempoNet
        self.TempoNet_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_head_Kt,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True)
        self.TempoNet_encoder = nn.TransformerEncoder(
            self.TempoNet_encoder_layer, num_layers=num_layers_Kt)

        # SpaNet
        self.SpaNet_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_head_Ks,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True)
        self.SpaNet_encoder = nn.TransformerEncoder(
            self.SpaNet_encoder_layer, num_layers=num_layers_Ks)

        # Encoder Embedding
        self.agent_embed = nn.Linear(agent_input_dim, d_model)
        self.lane_embed = LaneEmbeddingLayer(road_input_dim, d_model)

        self.PositionEncoding = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, d_model))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

        # Decoder

        # CrossAttender
        self.cross_depth = num_layers_Kc
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))

        # self.CrossAttender_decoderlayer = nn.TransformerDecoderLayer(
        #     d_model=d_model,
        #     nhead=num_head_Kc,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     activation=activation,
        #     batch_first=True,
        #     norm_first=True)
        # self.CrossAttender_decoder = nn.TransformerDecoder(
        #     self.CrossAttender_decoderlayer, num_layers=num_layers_Kc)

        self.agent_CrossAttender_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_head_Kc,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True
            ) for _ in range(self.cross_depth)
        ])

        self.road_CrossAttender_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_head_Kc,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True
            ) for _ in range(self.cross_depth)  # 同样创建 self.cross_depth 个独立的层
        ])

        # OutTrajectoryAndProbability
        self.pre_time = 60
        self.mlp_trajectory = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                            nn.ReLU(),
                                            nn.Linear(d_model * 2, 2 * self.pre_time))

        self.mlp_probability = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                             nn.ReLU(),
                                             nn.Linear(d_model * 2, 1))

        self.dense_predictor = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(
                256, self.pre_time * 2)
        )

    def forward(self, data):

        # ------------------Encoder--------------------
        x_agent_projection = self.agent_embed(data['x_src'])

        # x_agent_projection 形状: [batch, seq_a, Time, d_model]
        B, A, T, D = x_agent_projection.shape
        # x_agent_projection = x_agent_projection.view(B*A, T, -1)

        # 取反，True表示没有被mask的agent
        real_key_agent_mask = ~data["x_key_padding_mask"]  # [batch, seq_a]
        real_agent_feature = x_agent_projection[real_key_agent_mask]
        # real_agent_feature : [num_real_agent T D]

        # gather 关于real agent的时间掩码
        real_agent_time_mask = data["x_padding_mask"][...,
                                                      :50][real_key_agent_mask]
        # real_agent_time_mask : [num_real_agent T]

        x_agent_encode = self.TempoNet_encoder(
            src=real_agent_feature,
            src_key_padding_mask=real_agent_time_mask)

        x_agent_encode_full = torch.zeros(
            B, A, T, D, device=x_agent_projection.device, dtype=x_agent_projection.dtype)

        x_agent_encode_full[real_key_agent_mask] = x_agent_encode

        # x_agent_encode_full = torch.nan_to_num(x_agent_encode_full, nan=0.0)

        x_agent_maxpool = torch.max(x_agent_encode_full, dim=2).values
        # x_anget_maxpool : [batch seq_a d_model]

        # add position embedding
        # [B A 4]
        x_agent_maxpool += self.PositionEncoding(data["agent_pos_feat"])

        # add type embedding
        x_agent_maxpool += self.actor_type_embed[data["x_attr"][..., 2].long()]
        # x_anget_maxpool : [batch seq_a d_model]

        # road_process
        B, M, L, D = data["lane_src"].shape  # [batch num_L num_N 3]
        
        x_road_projection = self.lane_embed(
            data["lane_src"].view(-1, L, D).contiguous())
        x_road_projection = x_road_projection.view(B, M, -1)
        # add position embedding
        x_road_projection += self.PositionEncoding(data["road_pos_feat"])

        # add type embedding
        x_road_projection += self.lane_type_embed.repeat(B, M, 1)

        # concat road and agent
        x = torch.concat([x_agent_maxpool, x_road_projection], dim=1)

        spa_padding_mask = None
        if data["x_key_padding_mask"] is not None and data["lane_key_padding_mask"] is not None:
            spa_padding_mask = torch.cat(
                [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)

        # SpaNet
        encode_x = self.SpaNet_encoder(
            src=x, src_key_padding_mask=spa_padding_mask)
        
        encode_x = self.norm(encode_x)
        # ------------------Decoder--------------------

        # kv_angent = encode_x[:, 0].unsqueeze(1)
        # mask_agent = data["x_key_padding_mask"][:, 0].unsqueeze(1)

        batch_size = encode_x.size(0)
        batch_queries = self.queries.expand(batch_size, -1, -1)

        # x = self.CrossAttender_decoder(
        #     tgt=batch_queries,
        #     memory=encode_x,
        #     memory_key_padding_mask=spa_padding_mask
        # )

        for ali in range(self.cross_depth):
            batch_queries = self.agent_CrossAttender_decoder[ali](
                tgt=batch_queries,
                memory=encode_x[:, 0].unsqueeze(1),
                memory_key_padding_mask=data["x_key_padding_mask"][:, 0].unsqueeze(
                    1)
            )
            batch_queries = self.road_CrossAttender_decoder[ali](
                tgt=batch_queries,
                memory=encode_x[:, A:],
                memory_key_padding_mask=data["lane_key_padding_mask"]
            )

        y_hat = self.mlp_trajectory(batch_queries)
        pi = self.mlp_probability(batch_queries)

        B, N, _ = y_hat.shape
        y_hat = y_hat.view(B, N, self.pre_time, 2)

        x_others = encode_x[:, 1:A]
        y_hat_others = self.dense_predictor(
            x_others).view(B, -1, self.pre_time, 2)

        return {"y_hat": y_hat,
                "pi": pi,
                "y_hat_others": y_hat_others}
