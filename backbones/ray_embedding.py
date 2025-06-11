import spconv
import torch
import torch.nn as nn
import numpy as np
from .scn import SpMiddleResNetFHD
from ..registry import BACKBONES

class RayEmbedding(nn.Module):
    def __init__(self, freq_num=10, input_dim=3):
        super().__init__()
        self.freq_num = freq_num
        self.input_dim = input_dim  # 输入维度，适配体素特征

    def forward(self, points):
        # points: [M, 5]，假设前 3 维是 (x, y, z)，后 2 维是附加特征
        xyz = points[:, :3]  # [M, 3]
        dist = torch.norm(xyz, dim=-1, keepdim=True)  # [M, 1]
        direction = xyz / (dist + 1e-6)  # [M, 3]，避免除零
        # 生成频率
        freqs = 2. ** torch.linspace(0., self.freq_num-1, self.freq_num).to(points.device)  # [freq_num]

        # 为每个分量 (x, y, z) 构造频率组合
        direction_expanded = direction[:, :, None]  # [M, 3, 1]
        freqs_expanded = freqs[None, None, :]  # [1, 1, freq_num]
        encodings = direction_expanded * freqs_expanded  # [M, 3, freq_num]

        # 展平为 [M, 3 * freq_num]，按 x, y, z 顺序排列所有频率
        encodings = encodings.view(-1, 3 * self.freq_num)  # [M, 3 * freq_num]

        # 计算 sin 和 cos，并直接拼接为 [M, 3 * 2 * freq_num]
        dir_emb = torch.cat([torch.sin(encodings), torch.cos(encodings)], dim=-1)  # [M, 3 * 2 * freq_num]

        return dir_emb  # [M, 60]（freq_num=10 时）

@BACKBONES.register_module
class PointNetWithRayBackbone(nn.Module):
    def __init__(self, num_input_features=5, output_channels=256, ds_factor=8, norm_cfg=None, **kwargs):
        super(PointNetWithRayBackbone, self).__init__()
        self.num_input_features = num_input_features
        self.output_channels = output_channels
        self.ds_factor = ds_factor

        # 射线嵌入模块，基于体素中心点 (x, y, z)
        self.ray_emb = RayEmbedding(freq_num=10, input_dim=3)

        # 计算扩展后的输入特征维度：原始特征 (5) + RayEmbedding (60)
        extended_input_features = num_input_features + 60

        # 初始化 SpMiddleResNetFHD，输入特征维度为扩展后的值
        self.backbone = SpMiddleResNetFHD(
            num_input_features=extended_input_features,  # 修改为 65
            norm_cfg=norm_cfg,
            init_channel=16
        )

    def forward(self, voxel_feats, voxel_coords, batch_size, input_shape):
        # voxel_feats: [M, 5]，体素特征
        # voxel_coords: [M, 4]，体素坐标 (batch_idx, z, y, x)
        # batch_size: int，批次大小
        # input_shape: 输入尺寸

        # 生成 RayEmbedding 特征
        ray_feats = self.ray_emb(voxel_feats)  # [M, 60]
        
        # 将射线嵌入特征与输入体素特征拼接
        voxel_feats_with_ray = torch.cat([voxel_feats, ray_feats], dim=-1)  # [M, 5 + 60]

        # 输入 SpMiddleResNetFHD
        voxel_coords = voxel_coords.int()
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]
        ret = spconv.SparseConvTensor(voxel_feats_with_ray, voxel_coords, sparse_shape, batch_size)

        x = self.backbone.conv_input(ret)
        
        x_conv1 = self.backbone.conv1(x)
        x_conv2 = self.backbone.conv2(x_conv1)
        x_conv3 = self.backbone.conv3(x_conv2)
        x_conv4 = self.backbone.conv4(x_conv3)

        ret = self.backbone.extra_conv(x_conv4)

        ret = ret.dense()

        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features