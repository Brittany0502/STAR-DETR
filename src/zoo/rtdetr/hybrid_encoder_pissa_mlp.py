"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modified to include PiSSA adaptation for test-time domain adaptation
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import get_activation

from ...core import register
import pdb

__all__ = ['HybridEncoder', 'PiSSALayer', 'PiSSALinear']


class PiSSALayer(nn.Module):
    """PiSSA层实现 - 基于主奇异值分解的参数高效微调"""

    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # PiSSA参数 - 使用SVD初始化
        self.pissa_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.pissa_B = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.pissa_dropout = nn.Dropout(dropout)

        # 初始化PiSSA参数
        self._initialize_pissa()

    def _initialize_pissa(self):
        """使用PiSSA初始化方法"""
        # 使用正交初始化来模拟SVD的效果
        with torch.no_grad():
            # 初始化A矩阵 - 使用正交初始化
            nn.init.orthogonal_(self.pissa_A)
            # 初始化B矩阵 - 使用正交初始化
            nn.init.orthogonal_(self.pissa_B)

            # 缩放初始化
            self.pissa_A.data *= math.sqrt(1.0 / self.rank)
            self.pissa_B.data *= math.sqrt(1.0 / self.rank)

    def forward(self, x):
        # PiSSA前向传播: x @ A.T @ B.T * scaling
        pissa_out = (x @ self.pissa_A.T @ self.pissa_B.T) * self.scaling
        return self.pissa_dropout(pissa_out)


class PiSSALinear(nn.Module):
    """带PiSSA的线性层"""

    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.1, enable_pissa=True):
        super().__init__()

        self.enable_pissa = enable_pissa

        self.pissa = PiSSALayer(
            in_features,
            out_features,
            rank, alpha, dropout
        )

    def forward(self, x, linear):
        result = linear(x)
        if self.enable_pissa:
            result = result + self.pissa(x)
        return result


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 # PiSSA参数
                 enable_pissa=False,
                 pissa_rank=8,
                 pissa_alpha=16,
                 pissa_dropout=0.1):
        super().__init__()
        self.normalize_before = normalize_before
        self.enable_pissa = enable_pissa

        # 原始多头注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        # 仅在MLP部分注入PiSSA
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        if enable_pissa:
            # 冻结原始参数
            self.freeze_op(self.linear1)
            self.freeze_op(self.linear2)
            self.linear1_pissa = PiSSALinear(d_model, dim_feedforward, pissa_rank, pissa_alpha, pissa_dropout,
                                             enable_pissa)
            self.linear2_pissa = PiSSALinear(dim_feedforward, d_model, pissa_rank, pissa_alpha, pissa_dropout,
                                             enable_pissa)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def freeze_op(self, op):
        for param in op.parameters():
            param.requires_grad = False

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)

        # 直接调用原生attention
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        if self.enable_pissa:
            src = self.linear2_pissa(self.dropout(self.activation(self.linear1_pissa(src, self.linear1))), self.linear2)
        else:
            src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src

    def enable_pissa_adaptation(self):
        # 启用MLP部分的PiSSA
        if hasattr(self, 'linear1_pissa'):
            self.linear1_pissa.enable_pissa = True
        if hasattr(self, 'linear2_pissa'):
            self.linear2_pissa.enable_pissa = True

    def disable_pissa_adaptation(self):
        # 禁用MLP部分的PiSSA
        if hasattr(self, 'linear1_pissa'):
            self.linear1_pissa.enable_pissa = False
        if hasattr(self, 'linear2_pissa'):
            self.linear2_pissa.enable_pissa = False


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def enable_pissa_adaptation(self):
        """启用所有层的PiSSA适配"""
        for layer in self.layers:
            if hasattr(layer, 'enable_pissa_adaptation'):
                layer.enable_pissa_adaptation()

    def disable_pissa_adaptation(self):
        """禁用所有层的PiSSA适配"""
        for layer in self.layers:
            if hasattr(layer, 'disable_pissa_adaptation'):
                layer.disable_pissa_adaptation()


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='v2',
                 # PiSSA参数
                 enable_pissa=True,
                 pissa_rank=8,
                 pissa_alpha=16,
                 pissa_dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        # -----------有关PiSSA的配置信息-----------
        self.enable_pissa = enable_pissa
        self.pissa_rank = pissa_rank
        self.pissa_alpha = pissa_alpha
        self.pissa_dropout = pissa_dropout

        # print(f"HybridEncoder received pissa_dropout: {pissa_dropout}")

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()

            self.input_proj.append(proj)

        # encoder transformer with PiSSA
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
            enable_pissa=enable_pissa,
            pissa_rank=pissa_rank,
            pissa_alpha=pissa_alpha,
            pissa_dropout=pissa_dropout
        )

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                # 推理时，eval_spatial_size不是None
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    # 推理时，也使用自适应位置编码
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs

    def enable_pissa_adaptation(self):
        """启用PiSSA适配"""
        for encoder in self.encoder:
            encoder.enable_pissa_adaptation()

    def disable_pissa_adaptation(self):
        """禁用PiSSA适配"""
        for encoder in self.encoder:
            encoder.disable_pissa_adaptation()

    def get_pissa_parameters(self):
        """获取所有PiSSA参数"""
        pissa_params = []
        for encoder in self.encoder:
            for layer in encoder.layers:
                if hasattr(layer, 'linear1_pissa'):
                    pissa_params.extend([
                        layer.linear1_pissa.pissa.pissa_A,
                        layer.linear1_pissa.pissa.pissa_B,
                        layer.linear2_pissa.pissa.pissa_A,
                        layer.linear2_pissa.pissa.pissa_B,
                    ])
        return pissa_params

    def count_pissa_parameters(self):
        """统计PiSSA参数数量"""
        return sum(p.numel() for p in self.get_pissa_parameters())

    def load_pissa_weights(self, pissa_path):
        """加载PiSSA权重"""
        checkpoint = torch.load(pissa_path, map_location='cpu')

        # 如果是从init_pissa.py生成的权重格式
        if 'pissa_init' in checkpoint:
            pissa_weights = checkpoint['pissa_init']
        else:
            pissa_weights = checkpoint

        # 加载权重到对应的PiSSA层
        for name, param in self.named_parameters():
            if 'SVD_Lora' in name and name in pissa_weights:
                param.data.copy_(pissa_weights[name])

    def save_pissa_weights(self, save_path):
        """保存PiSSA权重"""
        pissa_weights = {}
        for name, param in self.named_parameters():
            if 'SVD_Lora' in name:
                pissa_weights[name] = param.data.clone()

        torch.save(pissa_weights, save_path)