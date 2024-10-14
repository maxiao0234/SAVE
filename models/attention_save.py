import torch
import torch.nn as nn
import math
from typing import Union
import curves
from timm.layers import trunc_normal_


class SAVEConfig:
    """The SAVE Configurations. This module assigns SAVE mode to each group of vectors.

    Components:
        abs: SAVE mode for absolute position encoding.
        vectors: Combination of Q, K, and V for SAVE, e.g.: '' indicates not using
            SAVE, 'q' indicates only using SAVE for Q, and 'qkv' indicates using
            SAVE for all components.
        mode: Aggregation mode of SAVE.
        Param: Parameterized mode of SAVE.

    abs Options: [True, False]
    mode Options: ['sequence', 'extension', 'hilbert']
    param Options: ['single', 'base', 'gate']
    """
    abs = False
    vectors = 'qkv'
    mode = 'extension'
    param = 'base'


class SpatialAggregationVectorEncoding(nn.Module):
    """Base class for SAVE. This block enables vectors to
    aggregate with respect to the given SAVE mode.

    Input (vector tensor): Input tokens, Q, K, or V.
    Output (vector tensor): Aggregated vectors.

    Args:
        hw_shape: Shape of input except class token.
        num_heads: Number of attention heads.
        head_dim: Dimensions of each position to aggregate.
        skip: Whether to skip the class token. Default: None. The int value
            means skipping the first which. To skip class token, specify 1.
    """
    def __init__(self, save_cfg, hw_shape, num_heads=1, head_dim=1, skip: Union[None, int]=None):
        super().__init__()
        self.save_cfg = save_cfg if save_cfg is not None else SAVEConfig
        self.num_orders = 2 if save_cfg.mode == 'extension' else 1
        self.hw_shape = hw_shape
        self.grid_size = hw_shape[0]
        self.length = hw_shape[0] * hw_shape[1]
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.skip = skip

        anchor = [[i, i, j, k] for i in range(self.length) for j in range(head_dim) for k in range(num_heads)]
        self.register_buffer('anchor',
                             torch.sparse_coo_tensor(
                                 indices=torch.tensor(anchor).transpose(0, 1),
                                 values=torch.ones(len(anchor)).float(),
                                 size=[self.length, self.length, head_dim, num_heads],
                             ).to_dense())

        for order in range(1, self.num_orders + 1):
            indices, values, num_nodes = getattr(self, f"build_spatial_{save_cfg.mode}")()
            setattr(self, f'num_nodes_{order}', num_nodes)
            self.register_buffer(f'spatial_table_{order}',
                                 torch.sparse_coo_tensor(
                                     indices=torch.tensor(indices).transpose(0, 1),
                                     values=torch.tensor(values).float(),
                                     size=[self.length, self.length, num_nodes],
                                 ).to_dense())
            getattr(self, f"init_parameters_{save_cfg.param}")(order)

    def _rank2pos(self, rank):
        return rank // self.hw_shape[1], rank % self.hw_shape[1]

    def _pos2rank(self, h, w):
        return h * self.hw_shape[1] + w

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

    def _param_norm(self, param):
        if self.num_orders == 1:
            return param - torch.mean(param, dim=1, keepdim=True)
        else:
            return param

    def build_spatial_sequence(self):
        order = 1
        num_direction = 1
        step_length = 8

        # norm_scale = step_length * 2
        norm_scale = 1

        indices = []
        values = []

        dist = 1 / norm_scale
        for rank in range(self.length):
            for step in range(step_length):
                left = self._shift(rank, step, 'left')
                right = self._shift(rank, step, 'right')
                if self._if_agg(left):
                    indices.append([rank, left, step])
                    values.append(dist)
                if self._if_agg(right):
                    indices.append([rank, right, step + step_length])
                    values.append(dist)

        return indices, values, order, num_direction

    def build_spatial_extension(self):
        curve_length = self.hw_shape[0]
        num_points = 4
        num_nodes = 4 * num_points
        # norm_scale = curve_length
        norm_scale = 1

        indices = []
        values = []

        for rank in range(self.length):
            h, w = self._rank2pos(rank)
            for step in range(1, curve_length):
                interpolation = ((num_points - 1) * (step - 1)) / (curve_length - 1)
                start_index = int(interpolation)
                dist = math.exp((- step ** 2 / (curve_length ** 2)))
                start_dist = dist * (1 - (interpolation - start_index)) / norm_scale
                next_dist = dist * (interpolation - start_index) / norm_scale

                pos_t = self._shift(rank, step, 'top')
                pos_b = self._shift(rank, step, 'bottom')
                pos_l = self._shift(rank, step, 'left')
                pos_r = self._shift(rank, step, 'right')

                if self._if_agg(pos_t, w_min=w, w_max=w):
                    indices.append([rank, pos_t, num_points * 0 + start_index])
                    values.append(start_dist)
                    indices.append([rank, pos_t, num_points * 0 + start_index + 1])
                    values.append(next_dist)
                if self._if_agg(pos_b, w_min=w, w_max=w):
                    indices.append([rank, pos_b, num_points * 1 + start_index])
                    values.append(start_dist)
                    indices.append([rank, pos_b, num_points * 1 + start_index + 1])
                    values.append(next_dist)
                if self._if_agg(pos_l, h_min=h, h_max=h):
                    indices.append([rank, pos_l, num_points * 2 + start_index])
                    values.append(start_dist)
                    indices.append([rank, pos_l, num_points * 2 + start_index + 1])
                    values.append(next_dist)
                if self._if_agg(pos_r, h_min=h, h_max=h):
                    indices.append([rank, pos_r, num_points * 3 + start_index])
                    values.append(start_dist)
                    indices.append([rank, pos_r, num_points * 3 + start_index + 1])
                    values.append(next_dist)

        return indices, values, num_nodes

    def build_spatial_hilbert(self):
        curve_iteration = 2
        num_points = 8
        sub_length = 2 * (num_points - 1) + 1
        num_nodes = num_points * 4
        # norm_scale = curve_length
        norm_scale = 1

        locations_t = curves.hilbertCurve(curve_iteration, rotation='topReverse')
        locations_b = curves.hilbertCurve(curve_iteration, rotation='bottomReverse')
        locations_l = curves.hilbertCurve(curve_iteration, rotation='left')
        locations_r = curves.hilbertCurve(curve_iteration, rotation='right')

        indices = []
        values = []

        for rank in range(self.length):
            h, w = self._rank2pos(rank)
            for step in range(sub_length):
                start_index_step = (step + 2) // 2
                start_dist = (start_index_step - (step / 2)) / norm_scale
                next_index_step = (start_index_step + 1) if start_index_step < (num_points - 1) else start_index_step
                next_dist = (1 - (start_index_step - (step / 2))) / norm_scale

                pos_t_h = h + locations_t[step + 1][0] - locations_t[0][0]
                pos_t_w = w + locations_t[step + 1][1] - locations_t[0][1]
                pos_b_h = h + locations_b[step + 1][0] - locations_b[0][0]
                pos_b_w = w + locations_b[step + 1][1] - locations_b[0][1]
                pos_l_h = h + locations_l[step + 1][0] - locations_l[0][0]
                pos_l_w = w + locations_l[step + 1][1] - locations_l[0][1]
                pos_r_h = h + locations_r[step + 1][0] - locations_r[0][0]
                pos_r_w = w + locations_r[step + 1][1] - locations_r[0][1]
                pos_t = self._pos2rank(pos_t_h, pos_t_w)
                pos_b = self._pos2rank(pos_b_h, pos_b_w)
                pos_l = self._pos2rank(pos_l_h, pos_l_w)
                pos_r = self._pos2rank(pos_r_h, pos_r_w)

                if self.grid_size > pos_t_h >= 0 and self.grid_size > pos_t_w >= 0:
                    indices.append([rank, pos_t, start_index_step + num_points * 0])
                    values.append(start_dist)
                    indices.append([rank, pos_t, next_index_step + num_points * 0])
                    values.append(next_dist)
                if self.grid_size > pos_b_h >= 0 and self.grid_size > pos_b_w >= 0:
                    indices.append([rank, pos_b, start_index_step + num_points * 1])
                    values.append(start_dist)
                    indices.append([rank, pos_b, next_index_step + num_points * 1])
                    values.append(next_dist)
                if self.grid_size > pos_l_h >= 0 and self.grid_size > pos_l_w >= 0:
                    indices.append([rank, pos_l, start_index_step + num_points * 2])
                    values.append(start_dist)
                    indices.append([rank, pos_l, next_index_step + num_points * 2])
                    values.append(next_dist)
                if self.grid_size > pos_r_h >= 0 and self.grid_size > pos_r_w >= 0:
                    indices.append([rank, pos_r, start_index_step + num_points * 3])
                    values.append(start_dist)
                    indices.append([rank, pos_r, next_index_step + num_points * 3])
                    values.append(next_dist)

        return indices, values, num_nodes

    def init_parameters_single(self, order):
        param = nn.Parameter(torch.empty(1, getattr(self, f"num_nodes_{order}"), self.head_dim, self.num_heads))
        setattr(self, f'spatial_parameters_{order}', param)
        nn.init.kaiming_uniform_(getattr(self, f'spatial_parameters_{order}'), a=math.sqrt(5))

    def init_parameters_base(self, order):
        coordinates_x = nn.Parameter(torch.empty(self.hw_shape[0], 1, self.head_dim, self.num_heads))
        coordinates_y = nn.Parameter(torch.empty(1, self.hw_shape[1], self.head_dim, self.num_heads))
        param_pos = nn.Parameter(torch.empty(2, getattr(self, f"num_nodes_{order}"), self.head_dim, self.num_heads))
        setattr(self, f'coordinates_x_{order}', coordinates_x)
        setattr(self, f'coordinates_y_{order}', coordinates_y)
        setattr(self, f'spatial_parameters_pos_{order}', param_pos)
        nn.init.kaiming_uniform_(getattr(self, f'coordinates_x_{order}'), a=math.sqrt(5))
        nn.init.kaiming_uniform_(getattr(self, f'coordinates_y_{order}'), a=math.sqrt(5))
        nn.init.kaiming_uniform_(getattr(self, f'spatial_parameters_pos_{order}'), a=math.sqrt(5))

    def init_parameters_gate(self, order):
        self.init_parameters_base(order)
        proj_node = nn.Parameter(torch.empty(getattr(self, f"num_nodes_{order}"), getattr(self, f"num_nodes_{order}")))
        proj_dim = nn.Parameter(torch.empty(self.head_dim // 2, self.head_dim))
        setattr(self, f'proj_node_{order}', proj_node)
        setattr(self, f'proj_dim_{order}', proj_dim)
        setattr(self, f'norm_{order}', nn.LayerNorm(self.head_dim // 2))
        nn.init.kaiming_uniform_(getattr(self, f'proj_node_{order}'), a=math.sqrt(5))
        nn.init.kaiming_uniform_(getattr(self, f'proj_dim_{order}'), a=math.sqrt(5))

    def get_weights_single(self, order):
        param = getattr(self, f'spatial_parameters_{order}')
        param = self._param_norm(param)
        param = param.expand(self.length, -1, -1, -1)
        param = self._param_norm(param)
        return param

    def get_weights_base(self, order):
        param = getattr(self, f'spatial_parameters_pos_{order}')
        coordinates_x = getattr(self, f'coordinates_x_{order}').expand(-1, self.hw_shape[1], -1, -1)
        coordinates_y = getattr(self, f'coordinates_y_{order}').expand(self.hw_shape[0], -1, -1, -1)
        coordinates = torch.stack([coordinates_x, coordinates_y], dim=2)
        coordinates = coordinates.flatten(start_dim=0, end_dim=1)
        param = torch.einsum('p i c n, i a c n-> p a c n', coordinates, param)
        param = self._param_norm(param)
        return param

    def get_weights_gate(self, order):
        param = self.get_weights_base(order)
        proj_node = getattr(self, f'proj_node_{order}')
        proj_dim = getattr(self, f'proj_dim_{order}')
        param_chunks = param.chunk(2, dim=2)
        param_u = param_chunks[0]
        param_u = param_u.transpose(-1, -2)
        param_v = param_chunks[1]
        param_u = getattr(self, f'norm_{order}')(param_u)
        param_u = torch.einsum('p b n d, b a-> p a d n', param_u, proj_node)
        param = param_u * param_v
        param = torch.einsum('p a d n, d c-> p a c n', param, proj_dim)
        param = self._param_norm(param)
        return param

    def aggregation(self, x):
        for order in range(1, self.num_orders + 1):
            weights = getattr(self, f"get_weights_{self.save_cfg.param}")(order)
            if self.num_heads == 1:
                weights = weights.expand(-1, -1, -1, x.shape[1])
            trans_mat = torch.einsum('p q a, p a c n -> p q c n', getattr(self, f'spatial_table_{order}'),
                                     weights) + self.anchor
            x = torch.einsum('p q c n, b n q c -> b n p c', trans_mat, x).contiguous()
        return x

    def forward(self, x):
        if self.skip is None:
            x = self.aggregation(x.unsqueeze(1)).squeeze(1)
        elif self.skip == 0:
            x = self.aggregation(x)
        else:
            cls_vectors = x[:, :, :self.skip, :]
            img_vectors = x[:, :, self.skip:, :]
            img_vectors = self.aggregation(img_vectors)
            x = torch.cat([cls_vectors, img_vectors], dim=2)
        return x


if __name__ == "__main__":
    import numpy as np

    Save = SpatialAggregationVectorEncoding(
        SAVEConfig,
        hw_shape=(224 // 16, 224 // 16),
        num_heads=1,
        head_dim=1,
        skip=None
    )
    Save.eval()
    n_parameters = sum(p.numel() for p in Save.parameters() if p.requires_grad)
    print('Params', n_parameters)

    spatial_paras = torch.ones(Save.num_nodes_1).float()
    spatial_table = Save.spatial_table_1
    agg_mat = torch.einsum('p q a, a -> p q', spatial_table, spatial_paras).numpy()

    for i in range(14 * 14):
        agg_mat[i, i] = -1
    print(np.around(agg_mat[66].reshape((14, 14)), 1))

