import os
from typing import Tuple, List, Callable, Union
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple, LayerNorm
from timm.layers import use_fused_attn
from timm.models.pvt_v2 import PyramidVisionTransformerV2

from .attention_save import SpatialAggregationVectorEncoding, SAVEConfig


__all__ = ['save_pvt_v2_b0', 'save_pvt_v2_b1', 'save_pvt_v2_b2']


class SAVEPyramidVisionTransformerV2(PyramidVisionTransformerV2):
    def __init__(self,
                 depths=(3, 4, 6, 3),
                 embed_dims=(64, 128, 256, 512),
                 num_heads=(1, 2, 4, 8),
                 sizes=(56, 28, 14, 7),
                 sr_ratios=(8, 4, 2, 1),
                 linear=False,
                 qkv_bias=True,
                 drop_rate=0.,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 save_cfg=SAVEConfig,
                 **kwargs):
        super().__init__(depths=depths, embed_dims=embed_dims, sr_ratios=sr_ratios, linear=linear, num_heads=num_heads,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, proj_drop_rate=proj_drop_rate,
                         attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, **kwargs)

        for i in range(1, len(depths)):
            hw_shape = (sizes[i], sizes[i])
            dim_out = embed_dims[i]
            num_head = num_heads[i]
            sr_ratio = sr_ratios[i]
            for j in range(depths[i]):
                attn = Attention(
                    save_cfg,
                    hw_shape,
                    dim_out,
                    num_heads=num_head,
                    sr_ratio=sr_ratio,
                    linear_attn=linear,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop_rate,
                    proj_drop=proj_drop_rate,
                )
                self.stages[i].blocks[j].attn = attn


class Attention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            save_cfg,
            hw_shape,
            dim,
            num_heads=8,
            sr_ratio=1,
            linear_attn=False,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        for vector in 'qkv':
            if vector in save_cfg.vectors:
                save = SpatialAggregationVectorEncoding(save_cfg, hw_shape, num_heads, self.head_dim, 0)
            else:
                save = nn.Identity()
            setattr(self, f'sa_{vector}_e', save)

        if not linear_attn:
            self.pool = None
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
            else:
                self.sr = None
                self.norm = None
            self.act = None
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, feat_size: List[int]):
        B, N, C = x.shape
        H, W = feat_size
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        q = self.sa_q_e(q)

        if self.pool is not None:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = self.act(x)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            if self.sr is not None:
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
                x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k, v = self.sa_k_e(k), self.sa_v_e(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@register_model
def save_pvt_v2_b0(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                   **kwargs) -> PyramidVisionTransformerV2:
    model = SAVEPyramidVisionTransformerV2(depths=(2, 2, 2, 2),
                                           embed_dims=(32, 64, 160, 256),
                                           num_heads=(1, 2, 5, 8), **kwargs)
    return model


@register_model
def save_pvt_v2_b1(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                   **kwargs) -> PyramidVisionTransformerV2:
    model = SAVEPyramidVisionTransformerV2(depths=(2, 2, 2, 2),
                                           embed_dims=(64, 128, 320, 512),
                                           num_heads=(1, 2, 5, 8), **kwargs)
    return model


@register_model
def save_pvt_v2_b2(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                   **kwargs) -> PyramidVisionTransformerV2:
    model = SAVEPyramidVisionTransformerV2(depths=(3, 4, 6, 3),
                                           embed_dims=(64, 128, 320, 512),
                                           num_heads=(1, 2, 5, 8), **kwargs)
    return model
