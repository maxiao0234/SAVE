import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.vision_transformer import VisionTransformer, _cfg
import numpy as np
import math

from typing import Union
import curves


__all__ = ['save_deit_t16_224', 'save_deit_s16_224', 'save_deit_b16_224']


PosEncoding = True  # Whether to use absolute position encoding. Options: [True, False]


class SAVEConfig:
    """The SAVE Configurations. This module assigns SAVE mode to each group of vectors.

    Components:
        ABS: SAVE mode for absolute position encoding.
        Q: SAVE mode for Q.
        K: SAVE mode for K.
        V: SAVE mode for V.

    Mode Options: [None, 'sequence', 'extension', 'hilbert', 'extension_augment', 'hilbert_augment']
    """
    ABS = None
    Q = 'hilbert_augment'
    K = None
    V = None


class SAVEDeiT(VisionTransformer):
    """The DeiT backbone using SAVE
    """
    def __init__(self, img_size=224, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12, qkv_bias=True, attn_drop_rate=0.1,
                 drop_rate=0., **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                         depth=depth, num_heads=num_heads, qkv_bias=qkv_bias,
                         attn_drop_rate=attn_drop_rate, drop_rate=drop_rate, **kwargs)

        hw_shape = to_2tuple(img_size // patch_size)

        if not PosEncoding:
            self.pos_embed = None

        self.sa_abs_e = SpatialAggregationVectorEncoding(hw_shape, SAVEConfig.ABS, embed_dim)

        for i in range(depth):
            self.blocks[i].attn = Attention(hw_shape,
                                            dim=embed_dim,
                                            num_heads=num_heads,
                                            qkv_bias=qkv_bias,
                                            attn_drop=attn_drop_rate,
                                            proj_drop=drop_rate)

    def forward_features(self, x):
        x = self.patch_embed(x)

        x = self.sa_abs_e(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        if self.pos_embed is not None:
            x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]


class Attention(nn.Module):
    def __init__(self, hw_shape, dim=1, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sa_q_e = SpatialAggregationVectorEncoding(hw_shape, SAVEConfig.Q, head_dim, skip=1, separable=False)
        self.sa_k_e = SpatialAggregationVectorEncoding(hw_shape, SAVEConfig.K, head_dim, skip=1, separable=False)
        self.sa_v_e = SpatialAggregationVectorEncoding(hw_shape, SAVEConfig.V, head_dim, skip=1, separable=False)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.sa_q_e(q)
        k = self.sa_k_e(k)
        v = self.sa_v_e(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialAggregationVectorEncoding(nn.Module):
    """Base class for SAVE. This block enables vectors to
    aggregate with respect to the given SAVE mode.

    Input (vector tensor): Input tokens, Q, K, or V.
    Output (vector tensor): Aggregated vectors.

    Args:
        hw_shape: Shape of input except class token.
        mode: Config for SAVE mode. Default: None, which means not using SAVE.
        dim: Dimensions of each position to aggregate.
        separable: Whether to use different parameters for each head.
        skip: Whether to skip the class token. Default: None. The int value
            means skipping the first which. To skip class token, specify 1.
    """
    def __init__(self, hw_shape, mode=None, dim=1, separable=True, skip: Union[None, int]=None):
        super().__init__()
        self.hw_shape = hw_shape
        self.length = hw_shape[0] * hw_shape[1]
        self.mode = mode
        self.dim = dim
        self.separable = separable
        self.skip = skip

        if mode is not None:
            indices, values, weight_length, order = getattr(self, f"_spatial_{mode}")()
            self.register_buffer('spatial_table',
                                 torch.sparse_coo_tensor(indices=torch.tensor(indices).transpose(0, 1),
                                                         values=torch.tensor(values).float(),
                                                         size=[self.length, self.length, weight_length],
                                                         ).to_dense())

            anchor_indices = [[i, i, j] for i in range(self.length) for j in range(dim)]
            self.register_buffer('anchor_weights',
                                 torch.sparse_coo_tensor(indices=torch.tensor(anchor_indices).transpose(0, 1),
                                                         values=torch.ones(len(anchor_indices)).float(),
                                                         size=[self.length, self.length, dim],
                                                         ).to_dense())

            self.order = order
            self.weight_length = weight_length
            for i in range(order):
                setattr(self, f'spatial_parameters_{i}', nn.Parameter(torch.empty(weight_length, dim)))
                trunc_normal_(getattr(self, f'spatial_parameters_{i}'))

    def _rank2pos(self, rank):
        return rank // self.hw_shape[1], rank % self.hw_shape[1]

    def _shift(self, rank, stride, direction):
        if 'left' in direction:
            rank = rank - stride
        if 'right' in direction:
            rank = rank + stride
        if 'top' in direction:
            rank = rank - stride * self.hw_shape[1]
        if 'bottom' in direction:
            rank = rank + stride * self.hw_shape[1]
        return rank

    def _reflect(self, rank, p, direction):
        assert direction in ['left', 'right', 'top', 'bottom']
        h, w = self._rank2pos(rank)
        h_, w_ = self._rank2pos(p)

        if 'left' == direction:
            assert p <= rank
            w_ = w_ - (h - h_) * self.hw_shape[1]
            h_ = h
            if w_ < 0:
                reflect_size = (w + 1) * 2
                offset = (- w_) % reflect_size
                if offset == 0:
                    w_ = 0
                elif offset <= w + 1:
                    w_ = offset - 1
                else:
                    w_ = w - (offset - w) + 2

        if 'top' in direction:
            assert p <= rank
            # assert w == w_
            if h_ < 0:
                reflect_size = (h + 1) * 2
                offset = (- h_) % reflect_size
                if offset == 0:
                    h_ = 0
                elif offset <= h + 1:
                    h_ = offset - 1
                else:
                    h_ = h - (offset - h) + 2

        if 'right' in direction:
            assert p >= rank
            w_ = w_ + (h_ - h) * self.hw_shape[1]
            h_ = h
            if w_ > self.hw_shape[1] - 1:
                reflect_size = (self.hw_shape[1] - w) * 2
                offset = (w_ - (self.hw_shape[1] - 1)) % reflect_size
                if offset == 0:
                    w_ = self.hw_shape[1] - 1
                elif offset <= self.hw_shape[1] - w:
                    w_ = self.hw_shape[1] - offset
                else:
                    w_ = w + (offset - (self.hw_shape[1] - w)) - 1

        if 'bottom' in direction:
            assert p >= rank
            # assert w == w_
            if h_ > self.hw_shape[0] - 1:
                reflect_size = (self.hw_shape[0] - h) * 2
                offset = (h_ - (self.hw_shape[0] - 1)) % reflect_size
                if offset == 0:
                    h_ = self.hw_shape[0] - 1
                elif offset <= self.hw_shape[0] - h:
                    h_ = self.hw_shape[0] - offset
                else:
                    h_ = h + (offset - (self.hw_shape[0] - h)) - 1

        return h_ * self.hw_shape[1] + w_

    def _if_agg(self, rank, h_min=None, h_max=None, w_min=None, w_max=None):
        if rank < 0:
            return False
        if rank >= self.length:
            return False
        h, w = self._rank2pos(rank)
        if h_min is not None and h < h_min:
            return False
        if h_max is not None and h > h_max:
            return False
        if w_min is not None and w < w_min:
            return False
        if w_max is not None and w > w_max:
            return False
        return True

    def _spatial_sequence(self, num_stride=8):
        order = 1
        weight_length = num_stride * 2
        norm_scale = num_stride * 2

        indices = []
        values = []

        dist = 1 / norm_scale
        for rank in range(self.length):
            for stride in range(num_stride):
                left = self._shift(rank, stride, 'left')
                right = self._shift(rank, stride, 'right')
                if self._if_agg(left):
                    indices.append([rank, left, stride])
                    values.append(dist)
                if self._if_agg(right):
                    indices.append([rank, right, stride + num_stride])
                    values.append(dist)

        return indices, values, weight_length, order

    def _spatial_extension(self, num_stride=4):
        order = 2
        weight_length = 4
        norm_scale = num_stride * 4

        indices = []
        values = []

        for rank in range(self.length):
            h, w = self._rank2pos(rank)
            for stride in range(num_stride):
                dist = math.exp((- stride ** 2 / (num_stride ** 2))) / norm_scale
                top = self._shift(rank, stride, 'top')
                bottom = self._shift(rank, stride, 'bottom')
                left = self._shift(rank, stride, 'left')
                right = self._shift(rank, stride, 'right')

                # top = self._reflect(rank, top, 'top')
                # bottom = self._reflect(rank, bottom, 'bottom')
                # left = self._reflect(rank, left, 'left')
                # right = self._reflect(rank, right, 'right')

                if self._if_agg(top, w_min=w, w_max=w):
                    indices.append([rank, top, 0])
                    values.append(dist)
                if self._if_agg(bottom, w_min=w, w_max=w):
                    indices.append([rank, bottom, 1])
                    values.append(dist)
                if self._if_agg(left, h_min=h, h_max=h):
                    indices.append([rank, left, 2])
                    values.append(dist)
                if self._if_agg(right, h_min=h, h_max=h):
                    indices.append([rank, right, 3])
                    values.append(dist)

        return indices, values, weight_length, order

    def _spatial_extension_augment(self, num_stride=4):
        indices, values, weight_length, order = self._spatial_extension(num_stride)

        norm_scale = num_stride * 4
        dist = 1 / norm_scale
        for rank in range(self.length):
            h, w = self._rank2pos(rank)
            for stride in range(num_stride):
                top = self._shift(rank, stride, 'top')
                bottom = self._shift(rank, stride, 'bottom')
                left = self._shift(rank, stride, 'left')
                right = self._shift(rank, stride, 'right')

                # top = self._reflect(rank, top, 'top')
                # bottom = self._reflect(rank, bottom, 'bottom')
                # left = self._reflect(rank, left, 'left')
                # right = self._reflect(rank, right, 'right')

                if self._if_agg(top, w_min=w, w_max=w):
                    indices.append([rank, top, weight_length + stride + num_stride * 0])
                    values.append(dist)
                if self._if_agg(bottom, w_min=w, w_max=w):
                    indices.append([rank, bottom, weight_length + stride + num_stride * 1])
                    values.append(dist)
                if self._if_agg(left, h_min=h, h_max=h):
                    indices.append([rank, left, weight_length + stride + num_stride * 2])
                    values.append(dist)
                if self._if_agg(right, h_min=h, h_max=h):
                    indices.append([rank, right, weight_length + stride + num_stride * 3])
                    values.append(dist)

        weight_length = weight_length + num_stride * 4
        return indices, values, weight_length, order

    def _spatial_hilbert(self, curve_iteration=2):
        order = 1
        weight_length = 8

        locs_top = curves.hilbertCurve(curve_iteration, rotation='topReverse')
        locs_bottom = curves.hilbertCurve(curve_iteration, rotation='bottomReverse')
        locs_left = curves.hilbertCurve(curve_iteration, rotation='left')
        locs_right = curves.hilbertCurve(curve_iteration, rotation='right')
        curve_length = len(locs_top)
        norm_scale = curve_length * 4

        indices = []
        values = []

        for rank in range(self.length):
            h, w = self._rank2pos(rank)
            for stride in range(curve_length):
                dist = ((curve_length - stride) / curve_length) / norm_scale
                dist_revise = (stride / curve_length) / norm_scale

                top_h = h + locs_top[stride][1] - locs_top[0][1]
                top_w = w + locs_top[stride][0] - locs_top[0][0]
                bottom_h = h + locs_bottom[stride][1] - locs_bottom[0][1]
                bottom_w = w + locs_bottom[stride][0] - locs_bottom[0][0]
                left_h = h + locs_left[stride][1] - locs_left[0][1]
                left_w = w + locs_left[stride][0] - locs_left[0][0]
                right_h = h + locs_right[stride][1] - locs_right[0][1]
                right_w = w + locs_right[stride][0] - locs_right[0][0]

                top = top_h * self.hw_shape[1] + top_w
                bottom = bottom_h * self.hw_shape[1] + bottom_w
                left = left_h * self.hw_shape[1] + left_w
                right = right_h * self.hw_shape[1] + right_w

                if top_h < self.hw_shape[0] and top_h >= 0 and top_w < self.hw_shape[1] and top_w >= 0:
                    indices.append([rank, top, 0])
                    values.append(dist)
                    indices.append([rank, top, 1])
                    values.append(dist_revise)
                if bottom_h < self.hw_shape[0] and bottom_h >= 0 and bottom_w < self.hw_shape[1] and bottom_w >= 0:
                    indices.append([rank, bottom, 2])
                    values.append(dist)
                    indices.append([rank, bottom, 3])
                    values.append(dist_revise)
                if left_h < self.hw_shape[0] and left_h >= 0 and left_w < self.hw_shape[1] and left_w >= 0:
                    indices.append([rank, left, 4])
                    values.append(dist)
                    indices.append([rank, left, 5])
                    values.append(dist_revise)
                if right_h < self.hw_shape[0] and right_h >= 0 and right_w < self.hw_shape[1] and right_w >= 0:
                    indices.append([rank, right, 6])
                    values.append(dist)
                    indices.append([rank, right, 7])
                    values.append(dist_revise)

        return indices, values, weight_length, order

    def _spatial_hilbert_augment(self, curve_iteration=2):
        indices, values, weight_length, order = self._spatial_hilbert(curve_iteration)

        locs_top = curves.hilbertCurve(curve_iteration, rotation='topReverse')
        locs_bottom = curves.hilbertCurve(curve_iteration, rotation='bottomReverse')
        locs_left = curves.hilbertCurve(curve_iteration, rotation='left')
        locs_right = curves.hilbertCurve(curve_iteration, rotation='right')
        curve_length = len(locs_top)
        norm_scale = curve_length * 4

        dist = 1 / norm_scale
        for rank in range(self.length):
            h, w = self._rank2pos(rank)
            for stride in range(curve_length):
                top_h = h + locs_top[stride][1] - locs_top[0][1]
                top_w = w + locs_top[stride][0] - locs_top[0][0]
                left_h = h + locs_left[stride][1] - locs_left[0][1]
                left_w = w + locs_left[stride][0] - locs_left[0][0]
                right_h = h + locs_right[stride][1] - locs_right[0][1]
                right_w = w + locs_right[stride][0] - locs_right[0][0]
                bottom_h = h + locs_bottom[stride][1] - locs_bottom[0][1]
                bottom_w = w + locs_bottom[stride][0] - locs_bottom[0][0]

                top = top_h * self.hw_shape[1] + top_w
                bottom = bottom_h * self.hw_shape[1] + bottom_w
                left = left_h * self.hw_shape[1] + left_w
                right = right_h * self.hw_shape[1] + right_w

                if top_h < self.hw_shape[0] and top_h >= 0 and top_w < self.hw_shape[1] and top_w >= 0:
                    indices.append([rank, top, weight_length + stride + curve_length * 0])
                    values.append(dist)
                if bottom_h < self.hw_shape[0] and bottom_h >= 0 and bottom_w < self.hw_shape[1] and bottom_w >= 0:
                    indices.append([rank, bottom, weight_length + stride + curve_length * 1])
                    values.append(dist)
                if left_h < self.hw_shape[0] and left_h >= 0 and left_w < self.hw_shape[1] and left_w >= 0:
                    indices.append([rank, left, weight_length + stride + curve_length * 2])
                    values.append(dist)
                if right_h < self.hw_shape[0] and right_h >= 0 and right_w < self.hw_shape[1] and right_w >= 0:
                    indices.append([rank, right, weight_length + stride + curve_length * 3])
                    values.append(dist)

        weight_length = weight_length + curve_length * 4
        return indices, values, weight_length, order

    def aggregation(self, x):
        for i in range(self.order):
            spatial_parameters = getattr(self, f'spatial_parameters_{i}')
            trans_mat = torch.einsum('p q n, n k -> p q k', self.spatial_table, spatial_parameters) + self.anchor_weights
            x = torch.einsum('p q k, b q k -> b p k', trans_mat, x)
        return x

    def forward(self, x):
        if self.mode is not None:
            if self.skip is not None:
                B, N, L, C = x.shape
                if self.separable:
                    x = x.permute(0, 2, 1, 3).reshape(B, L, N * C).contiguous()
                    cls_vectors = x[:, :self.skip, :]
                    img_vectors = x[:, self.skip:, :]
                    img_vectors = self.aggregation(img_vectors)
                    x = torch.cat([cls_vectors, img_vectors], dim=1)
                    x = x.reshape(B, L, N, C).permute(0, 2, 1, 3).contiguous()
                else:
                    x = x.reshape(B * N, L, C).contiguous()
                    cls_vectors = x[:, :self.skip, :]
                    img_vectors = x[:, self.skip:, :]
                    img_vectors = self.aggregation(img_vectors)
                    x = torch.cat([cls_vectors, img_vectors], dim=1)
                    x = x.reshape(B, N, L, C).contiguous()
            else:
                x = self.aggregation(x)

        return x


@register_model
def save_deit_t16_224(pretrained=False, **kwargs):
    model = SAVEDeiT(img_size=224,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def save_deit_s16_224(pretrained=False, **kwargs):
    model = SAVEDeiT(img_size=224,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def save_deit_b16_224(pretrained=False, **kwargs):
    model = SAVEDeiT(img_size=224,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == "__main__":
    import argparse
    from main import get_args_parser

    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    Model = save_deit_t16_224()
    n_parameters = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters} ({n_parameters / (1024 * 1024):.2f}M)')

    x = torch.rand((2, 3, 224, 224))
    out = Model(x)
    print(out.shape)

    spatial_paras = torch.ones((Model.sa_abs_e.weight_length)).float()
    spatial_table = Model.sa_abs_e.spatial_table
    agg_mat = torch.einsum('p q n, n -> p q', spatial_table, spatial_paras)
    agg_mat = np.around(agg_mat.numpy(), 1)

    for i in range(14 * 14):
        agg_mat[i, i] = -1

    print(agg_mat[46].reshape((14, 14)))


