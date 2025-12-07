from typing import Optional

from sympy import asec, false
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor, layer_norm


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        linear_bias=True,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features,bias=linear_bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features,bias=linear_bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias,
        linear_bias=True,
        drop=0,
        drop_path=0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cross_attn=False
    ):
        super().__init__()

        self.cross_attn = cross_attn
        if self.cross_attn:
            self.normk = norm_layer(dim)
            self.normv = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            bias=qkv_bias,
            batch_first=True
        )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            linear_bias = linear_bias,
            drop=drop,
        )
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_custom(
        self,
        src,
        key_padding_mask: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
    ):
        assert k is not None and v is not None
        q = self.norm1(src)
        k = self.normk(k)
        v = self.normv(v)

        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]

        src = q + self.drop_path1(attn_output)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))

        return src

    def forward_pre(
        self,
        src,
        key_padding_mask: Optional[Tensor] = None,
    ):

        norm_src = self.norm1(src)
        attn_output = self.attn(
            query=norm_src,
            key=norm_src,
            value=norm_src,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]
        src = src + self.drop_path1(attn_output)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))

        return src

    def forward(
        self,
        src,
        key_padding_mask: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
    ):

        if self.cross_attn:
            return self.forward_custom(src=src, key_padding_mask=key_padding_mask, k=k, v=v)
        else:
            return self.forward_pre(src=src,key_padding_mask=key_padding_mask)
