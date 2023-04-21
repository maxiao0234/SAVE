import os
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
        Vectors: Combination of Q, K, and V for SAVE, e.g.: '' indicates not using
            SAVE, 'q' indicates only using SAVE for Q, and 'qkv' indicates using
            SAVE for all components.
        Mode: Aggregation mode of SAVE.
        Param: Parameterized mode of SAVE.

    ABS Options: [True, False]
    Mode Options: ['sequence', 'extension', 'hilbert']
    Param Options: ['single', 'composite', 'mlp']
    """
    ABS = False
    Vectors = 'qkv'
    Mode = 'hilbert'
    Param = 'mlp'


class SAVEDeiT(VisionTransformer):
    """The DeiT backbone using SAVE
    """
    def __init__(self, img_size=224, patch_size=16,
                 embed_dim=768, depth=12, num_heads=12, qkv_bias=True, attn_drop_rate=0.1,
                 drop_rate=0., pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                         depth=depth, num_heads=num_heads, qkv_bias=qkv_bias,
                         attn_drop_rate=attn_drop_rate, drop_rate=drop_rate, **kwargs)

        hw_shape = to_2tuple(img_size // patch_size)

        if not PosEncoding:
            self.pos_embed = None

        self.sa_abs_e = SpatialAggregationVectorEncoding(hw_shape, SAVEConfig.ABS, 1, embed_dim, None)

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
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, hw_shape, dim, num_heads, qkv_bias=False, attn_drop=0.1, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        for vector in 'qkv':
            enable = True if vector in SAVEConfig.Vectors else False
            save = SpatialAggregationVectorEncoding(hw_shape, enable, num_heads, head_dim, 1)
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


class SpatialAggregationVectorEncoding(nn.Module):
    """Base class for SAVE. This block enables vectors to
    aggregate with respect to the given SAVE mode.

    Input (vector tensor): Input tokens, Q, K, or V.
    Output (vector tensor): Aggregated vectors.

    Args:
        hw_shape: Shape of input except class token.
        enable: Whether to use SAVE. Default: None, which means not using SAVE.
        num_heads: Number of attention heads.
        head_dim: Dimensions of each position to aggregate.
        skip: Whether to skip the class token. Default: None. The int value
            means skipping the first which. To skip class token, specify 1.
    """
    def __init__(self, hw_shape, enable=True, num_heads=1, head_dim=1, skip: Union[None, int]=None):
        super().__init__()
        self.hw_shape = hw_shape
        self.enable = enable
        self.length = hw_shape[0] * hw_shape[1]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.skip = skip

        if enable:
            anchor = [[i, i, j, k] for i in range(self.length) for j in range(num_heads) for k in range(head_dim)]
            self.register_buffer('anchor',
                                 torch.sparse_coo_tensor(
                                     indices=torch.tensor(anchor).transpose(0, 1),
                                     values=torch.ones(len(anchor)).float(),
                                     size=[self.length, self.length, num_heads, head_dim],
                                 ).to_dense())

            indices, values, order, num_direction, num_entry = getattr(self, f"_spatial_{SAVEConfig.Mode}")()

            self.indices = indices
            self.order = order
            self.num_direction = num_direction
            self.num_entry = num_entry
            self.register_buffer('spatial_table',
                                 torch.sparse_coo_tensor(
                                     indices=torch.tensor(indices).transpose(0, 1),
                                     values=torch.tensor(values).float(),
                                     size=[self.length, self.length, num_direction, num_entry],
                                 ).to_dense().reshape(self.length, self.length, num_direction * num_entry))

            getattr(self, f"_init_parameters_{SAVEConfig.Param}")()

    def _rank2pos(self, rank):
        return rank // self.hw_shape[1], rank % self.hw_shape[1]

    def _pos2rank(self, h, w):
        return h * self.hw_shape[1] + w

    def _get_central_coefficient(self, h, w):
        half_h = self.hw_shape[0] // 2
        half_w = self.hw_shape[1] // 2
        adjustment = 0.5
        m_ij = (self.hw_shape[0] * adjustment) ** 2 + (self.hw_shape[1] * adjustment) ** 2
        if h >= half_h:
            m_i = (self.hw_shape[0] - h - 1 - half_h) ** 2
        else:
            m_i = (h - half_h) ** 2
        if w >= half_w:
            m_j = (self.hw_shape[1] - w - 1 - half_w) ** 2
        else:
            m_j = (w - half_w) ** 2

        return math.exp((- (m_i + m_j) / m_ij))

    def _get_coordinates(self):
        rank_list = []
        coordinates = []

        for item in self.indices:
            rank = item[0]
            if rank not in rank_list:
                h, w = self._rank2pos(rank)
                half_h = self.hw_shape[0] // 2
                half_w = self.hw_shape[1] // 2
                h_f = h / half_h
                w_f = w / half_w
                h_r = (self.hw_shape[0] - h - 1) / half_h
                w_r = (self.hw_shape[1] - w - 1) / half_w
                coordinates.append([h_f, w_f, h_r, w_r])
                rank_list.append(rank)

        return coordinates

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

    def _spatial_sequence(self, step_length=8):
        order = 1
        num_direction = 1
        num_entry = step_length * 2

        norm_scale = 1

        indices = []
        values = []

        dist = 1 / norm_scale
        for rank in range(self.length):
            for step in range(step_length):
                left = self._shift(rank, step, 'left')
                right = self._shift(rank, step, 'right')
                if self._if_agg(left):
                    indices.append([rank, left, 0, step])
                    values.append(dist)
                if self._if_agg(right):
                    indices.append([rank, right, 0, step + step_length])
                    values.append(dist)

        return indices, values, order, num_direction, num_entry

    def _spatial_extension(self, step_length=4):
        order = 2
        num_direction = 4
        num_entry = step_length - 1

        # norm_scale = sum([math.exp((- stride ** 2 / (num_stride ** 2))) for stride in range(num_stride)])
        norm_scale = 1

        indices = []
        values = []

        for rank in range(self.length):
            h, w = self._rank2pos(rank)
            for step in range(1, step_length):
                dist = math.exp((- step ** 2 / (step_length ** 2))) / norm_scale
                step_entry = step - 1

                top = self._shift(rank, step, 'top')
                bottom = self._shift(rank, step, 'bottom')
                left = self._shift(rank, step, 'left')
                right = self._shift(rank, step, 'right')

                if self._if_agg(top, w_min=w, w_max=w):
                    indices.append([rank, top, 0, step_entry])
                    values.append(dist)
                if self._if_agg(bottom, w_min=w, w_max=w):
                    indices.append([rank, bottom, 1, step_entry])
                    values.append(dist)
                if self._if_agg(left, h_min=h, h_max=h):
                    indices.append([rank, left, 2, step_entry])
                    values.append(dist)
                if self._if_agg(right, h_min=h, h_max=h):
                    indices.append([rank, right, 3, step_entry])
                    values.append(dist)

        return indices, values, order, num_direction, num_entry

    def _spatial_hilbert(self, curve_iteration=2):
        order = 1
        num_direction = 4
        curve_length = 4 ** curve_iteration
        num_entry = curve_length // 4

        # norm_scale = curve_length / 2
        norm_scale = 1

        locs_top = curves.hilbertCurve(curve_iteration, rotation='topReverse')
        locs_bottom = curves.hilbertCurve(curve_iteration, rotation='bottomReverse')
        locs_left = curves.hilbertCurve(curve_iteration, rotation='left')
        locs_right = curves.hilbertCurve(curve_iteration, rotation='right')

        indices = []
        values = []

        for rank in range(self.length):
            h, w = self._rank2pos(rank)
            for step in range(1, curve_length):
                dist = ((curve_length - step) / curve_length) / norm_scale
                dist_revise = (step / curve_length) / norm_scale
                step_entry = int(step * (num_entry / curve_length))

                top_h = h + locs_top[step][1] - locs_top[0][1]
                top_w = w + locs_top[step][0] - locs_top[0][0]
                bottom_h = h + locs_bottom[step][1] - locs_bottom[0][1]
                bottom_w = w + locs_bottom[step][0] - locs_bottom[0][0]
                left_h = h + locs_left[step][1] - locs_left[0][1]
                left_w = w + locs_left[step][0] - locs_left[0][0]
                right_h = h + locs_right[step][1] - locs_right[0][1]
                right_w = w + locs_right[step][0] - locs_right[0][0]

                top = top_h * self.hw_shape[1] + top_w
                bottom = bottom_h * self.hw_shape[1] + bottom_w
                left = left_h * self.hw_shape[1] + left_w
                right = right_h * self.hw_shape[1] + right_w

                if self.hw_shape[0] > top_h >= 0 and self.hw_shape[1] > top_w >= 0:
                    indices.append([rank, top, 0, step_entry])
                    values.append(dist)
                    indices.append([rank, top, 1, step_entry])
                    values.append(dist_revise)
                if self.hw_shape[0] > bottom_h >= 0 and self.hw_shape[1] > bottom_w >= 0:
                    indices.append([rank, bottom, 1, step_entry])
                    values.append(dist)
                    indices.append([rank, bottom, 0, step_entry])
                    values.append(dist_revise)
                if self.hw_shape[0] > left_h >= 0 and self.hw_shape[1] > left_w >= 0:
                    indices.append([rank, left, 2, step_entry])
                    values.append(dist)
                    indices.append([rank, left, 3, step_entry])
                    values.append(dist_revise)
                if self.hw_shape[0] > right_h >= 0 and self.hw_shape[1] > right_w >= 0:
                    indices.append([rank, right, 3, step_entry])
                    values.append(dist)
                    indices.append([rank, right, 2, step_entry])
                    values.append(dist_revise)

        return indices, values, order, num_direction, num_entry

    def _init_parameters_single(self):
        for i in range(self.order):
            param = nn.Parameter(torch.zeros(self.num_direction, 1, self.num_heads, self.head_dim))
            setattr(self, f'spatial_parameters_{i}', param)
            nn.init.kaiming_uniform_(getattr(self, f'spatial_parameters_{i}'), a=math.sqrt(5))

    def _init_parameters_composite(self):
        for i in range(self.order):
            param = nn.Parameter(torch.zeros(self.num_direction, self.num_entry, self.num_heads, self.head_dim))
            setattr(self, f'spatial_parameters_{i}', param)
            nn.init.kaiming_uniform_(getattr(self, f'spatial_parameters_{i}'), a=math.sqrt(5))

    def _init_parameters_mlp(self):
        hidden = self.head_dim // 16
        # weight_length = self.num_direction
        weight_length = self.num_direction * self.num_entry
        for i in range(self.order):
            setattr(self, f'w_1_{i}', nn.Parameter(torch.ones(4, hidden, self.num_heads)))
            setattr(self, f'w_2_{i}', nn.Parameter(torch.ones(weight_length, hidden, self.num_heads, self.head_dim)))
            setattr(self, f'b_1_{i}', nn.Parameter(torch.ones(1, 1, self.num_heads)))
            setattr(self, f'b_2_{i}', nn.Parameter(torch.ones(1, 1, self.num_heads, 1)))
            setattr(self, f'act_{i}', nn.ReLU())
            nn.init.kaiming_uniform_(getattr(self, f'w_1_{i}'), a=math.sqrt(5))
            nn.init.kaiming_uniform_(getattr(self, f'w_2_{i}'), a=math.sqrt(5))
            trunc_normal_(getattr(self, f'b_1_{i}'), std=.02)
            trunc_normal_(getattr(self, f'b_2_{i}'), std=.02)
        coordinates = self._get_coordinates()
        self.register_buffer('coordinates', torch.tensor(coordinates).float())

    def _get_parameters_single(self, order):
        param = getattr(self, f'spatial_parameters_{order}')
        param = param.expand(-1, self.num_entry, -1, -1)
        param = param.reshape(self.num_direction * self.num_entry, self.num_heads, self.head_dim)
        param = param.unsqueeze(0).expand(self.length, -1, -1, -1)
        return param

    def _get_parameters_composite(self, order):
        param = getattr(self, f'spatial_parameters_{order}')
        param = param.reshape(self.num_direction * self.num_entry, self.num_heads, self.head_dim)
        param = param.unsqueeze(0).expand(self.length, -1, -1, -1)
        return param

    def _get_parameters_mlp(self, order):
        w_1 = getattr(self, f'w_1_{order}')
        b_1 = getattr(self, f'b_1_{order}')
        act = getattr(self, f'act_{order}')
        w_2 = getattr(self, f'w_2_{order}')
        b_2 = getattr(self, f'b_2_{order}')

        param = torch.einsum('p i, i d n -> p d n', self.coordinates, w_1) + b_1
        param = act(param)
        param = torch.einsum('p d n, a d n c -> p a n c', param, w_2) + b_2
        return param

    def aggregation(self, x):
        for i in range(self.order):
            param = getattr(self, f"_get_parameters_{SAVEConfig.Param}")(i)
            if self.num_heads == 1:
                B, N, L, C = x.shape
                param = param.expand(-1, -1, N, -1)
            trans_mat = torch.einsum('p q a, p a n c -> p q n c', self.spatial_table, param) + self.anchor
            x = torch.einsum('p q n c, b n q c -> b n p c', trans_mat, x)
        return x

    def forward(self, x):
        if self.enable:
            if self.skip is not None:
                cls_vectors = x[:, :, :self.skip, :]
                img_vectors = x[:, :, self.skip:, :]
                img_vectors = self.aggregation(img_vectors)
                x = torch.cat([cls_vectors, img_vectors], dim=2)
            else:
                x = self.aggregation(x.unsqueeze(1)).squeeze(1)
        return x


@register_model
def save_deit_t16_224(pretrained=False, **kwargs):
    model = SAVEDeiT(
        # img_size=224,
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
    Model = save_deit_t16_224()
    Model.eval()
    n_parameters = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters} ({n_parameters / (1024 * 1024):.2f}M)')

    x = torch.rand((2, 3, 224, 224))
    out = Model(x)
    print(out.shape)

    spatial_paras = torch.ones((Model.sa_abs_e.num_direction * Model.sa_abs_e.num_entry)).float()
    spatial_table = Model.sa_abs_e.spatial_table
    agg_mat = torch.einsum('p q a, a -> p q', spatial_table, spatial_paras).numpy()

    # for i in range(14 * 14):
    #     agg_mat[i, i] = -1

    print(np.around(agg_mat[60].reshape((14, 14)), 1))



