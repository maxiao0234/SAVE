import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.registry import register_model
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import VisionTransformer, _cfg
import numpy as np

from .attention_save import SpatialAggregationVectorEncoding, SAVEConfig


__all__ = ['save_deit_t16_224', 'save_deit_s16_224', 'save_deit_b16_224']


PosEncoding = True  # Whether to use absolute position encoding. Options: [True, False]


class SAVEDeiT(VisionTransformer):
    """The DeiT backbone using SAVE
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                 qkv_bias=True, attn_drop_rate=0.1, drop_rate=0.,
                 pretrained_cfg=None, pretrained_cfg_overlay=None,
                 save_cfg=SAVEConfig, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                         depth=depth, num_heads=num_heads, qkv_bias=qkv_bias,
                         attn_drop_rate=attn_drop_rate, drop_rate=drop_rate, **kwargs)

        hw_shape = to_2tuple(img_size // patch_size)

        if not PosEncoding:
            self.pos_embed = None

        if save_cfg.abs:
            self.sa_abs_e = SpatialAggregationVectorEncoding(save_cfg, hw_shape, 2, 1, embed_dim, None)
        else:
            self.sa_abs_e = nn.Identity()

        for i in range(depth):
            self.blocks[i].attn = Attention(save_cfg,
                                            hw_shape,
                                            dim=embed_dim,
                                            num_heads=num_heads,
                                            qkv_bias=qkv_bias,
                                            attn_drop=attn_drop_rate,
                                            proj_drop=drop_rate)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.sa_abs_e(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, save_cfg, hw_shape, dim, num_heads, qkv_bias=False, attn_drop=0.1, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        for vector in 'qkv':
            if vector in save_cfg.vectors:
                save = SpatialAggregationVectorEncoding(save_cfg, hw_shape, num_heads, head_dim, 1)
            else:
                save = nn.Identity()
            setattr(self, f'sa_{vector}_e', save)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k, v = self.sa_q_e(q), self.sa_k_e(k), self.sa_v_e(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@register_model
def save_deit_t16_224(pretrained=False, **kwargs):
    model = SAVEDeiT(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3,
                     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def save_deit_s16_224(pretrained=False, **kwargs):
    model = SAVEDeiT(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
                     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def save_deit_b16_224(pretrained=False, **kwargs):
    model = SAVEDeiT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

