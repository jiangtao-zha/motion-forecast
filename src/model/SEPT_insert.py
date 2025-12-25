from matplotlib import axis
import torch
import torch.nn as nn
from model.lane_embedding import LaneEmbeddingLayer
from model.transformer_blocks import Block

class SEPT(nn.Module):
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
                 mlp_ratio: float,
                 qkv_bias:bool,
                 linear_bias:bool,
                 drop_path:float,
                 dropout: float = 0,
                 activation=nn.GELU):
        super().__init__()
        # Encoder

        # TempoNet
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers_Kt)]
        self.TempoNet_encoder = nn.ModuleList(
            Block(dim=d_model,
                  num_heads=num_head_Kt,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  linear_bias=linear_bias,
                  drop=dropout,
                  drop_path=dpr[i],
                  cross_attn=False
                  )
            for i in range(num_layers_Kt)
        )

        # SpaNet
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers_Ks)]
        self.SpaNet_encoder = nn.ModuleList(
            Block(dim=d_model,
                  num_heads=num_head_Ks,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  linear_bias=linear_bias,
                  drop=dropout,
                  drop_path=dpr[i],
                  cross_attn=False
                  )
            for i in range(num_layers_Ks)
        )

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

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.cross_depth)]

        self.agent_CrossAttender_decoder = nn.ModuleList(
            Block(dim=d_model,
                  num_heads=num_head_Kc,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  linear_bias=linear_bias,
                  drop=dropout,
                  drop_path=dpr[i],
                  cross_attn=True
                  )
            for i in range(self.cross_depth)
        )

        self.road_CrossAttender_decoder = nn.ModuleList(
            Block(dim=d_model,
                  num_heads=num_head_Kc,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  linear_bias=linear_bias,
                  drop=dropout,
                  drop_path=dpr[i],
                  cross_attn=True
                  )
            for i in range(self.cross_depth)
        )

        # OutTrajectoryAndProbability
        self.pre_time = 60
        self.mlp_trajectory = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                            nn.ReLU(),
                                            nn.Linear(d_model * 2, 2 * self.pre_time))

        self.mlp_probability = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                             nn.ReLU(),
                                             nn.Linear(d_model * 2, 1)
                                             )

        self.dense_predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(
                d_model * 2, self.pre_time * 2)
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


        for blk in self.TempoNet_encoder:
            real_agent_feature = blk(src=real_agent_feature,key_padding_mask=real_agent_time_mask)
            
        x_agent_maxpool = torch.max(real_agent_feature,axis=1).values

        x_agent_encode_full = torch.zeros(
            B, A, D, device=x_agent_projection.device, dtype=x_agent_projection.dtype)

        x_agent_encode_full[real_key_agent_mask] = x_agent_maxpool

        # add position embedding
        # [B A 4]
        x_agent_encode_full += self.PositionEncoding(data["agent_pos_feat"])

        # add type embedding
        x_agent_encode_full += self.actor_type_embed[data["x_attr"][..., 2].long()]
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
        encode_x = torch.concat([x_agent_encode_full, x_road_projection], dim=1)

        spa_padding_mask = None
        if data["x_key_padding_mask"] is not None and data["lane_key_padding_mask"] is not None:
            spa_padding_mask = torch.cat(
                [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1)

        # SpaNet
        for blk in self.SpaNet_encoder:
            encode_x = blk(src=encode_x,key_padding_mask=spa_padding_mask)
        
        encode_x = self.norm(encode_x)
        target_feat = encode_x[:,0]
        # ------------------Decoder--------------------


        batch_size = encode_x.size(0)
        batch_queries = self.queries.expand(batch_size, -1, -1)

        for ali in range(self.cross_depth):
            batch_queries = self.agent_CrossAttender_decoder[ali](
                src=batch_queries,
                key_padding_mask=data["x_key_padding_mask"][:, 0].unsqueeze(1),
                k=encode_x[:, 0].unsqueeze(1),
                v=encode_x[:, 0].unsqueeze(1)
            )
            batch_queries = self.road_CrossAttender_decoder[ali](
                src=batch_queries,
                key_padding_mask=data["lane_key_padding_mask"],
                k=encode_x[:, A:],
                v=encode_x[:, A:]
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
                "y_hat_others": y_hat_others,
                "target_feat": target_feat}
