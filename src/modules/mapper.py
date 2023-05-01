from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import e2cnn
from e2cnn import gspaces
from e2cnn import nn as e2nn

from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor
from e2cnn.nn.modules.equivariant_module import EquivariantModule
# from .pos_encod import PositionalEncoding

from scripts.init import ex
from utils.tensor_transform import vis_to_conv2d, flatten_repr_channel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import autograd, Tensor
from collections import OrderedDict
import math


class NavMapper(nn.Module):
    def __init__(self, map_height, map_width, num_views, img_height, img_width, workspace_size=None):
        super().__init__()

        assert num_views == 4
        assert img_height == img_width

        self.map_height, self.map_width = map_height, map_width
        self.num_views = num_views
        self.img_height, self.img_width = img_height, img_width
        self.img_rgb = 3

        self.hidden_dim = [32, 64]
        self.img_embed_dim = 256
        self.map_dim = 1

        # step 1: use regular CNN to embed 4 images
        # input = [batch, map_height, map_width, (4, img_height, img_width, RGB)]
        # output = [batch, map_height, map_width, (4, img_embedding)]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(self.img_rgb, self.hidden_dim[0],
                      kernel_size=10, stride=4, padding=4),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[0], self.hidden_dim[1],
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(self.hidden_dim[1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.img_embed_dim,
                      kernel_size=4, stride=1),
            nn.BatchNorm2d(self.img_embed_dim),
            nn.ReLU(),
        )

        self.map_kernel_sizes = []

        # step 2: use regular CNN to process 4 embeddings again
        # input = [batch, map_height, map_width, (4, img_embedding)]
        # output = [batch, map_height, map_width, (obstacle_embedding)]
        self.map_encoder = nn.Sequential(
            nn.Conv2d(self.img_embed_dim * self.num_views, self.hidden_dim[0],
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[0], self.map_dim,
                      kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        batch_size, map_height, map_width, num_views, img_height, img_width, img_rgb = x.size()
        assert (
                   map_height, map_width, num_views, img_height, img_width, img_rgb
               ) == (
                   self.map_height, self.map_width, self.num_views, self.img_height, self.img_width, 3
               )

        # > flatten map dimensions, process individual images, shared across batch/locations/views
        x = vis_to_conv2d(x)
        # > size = (0) (batch_size * map_width * map_height * |C_4|) x [(1) RGB x (2) image_width x (3) image_height]

        # > step 1
        x = self.img_encoder(x)
        # size = (0) (batch_size * map_width * map_height) x (1) |C_4| x (2) image_embedding

        # > reshape for processing over map, in two steps
        x = x.view(batch_size, num_views, map_height, map_width, self.img_embed_dim)
        x = x.view(batch_size, num_views * self.img_embed_dim, map_height, map_width)

        # > step 2
        x = self.map_encoder(x)
        x = x.view(batch_size, self.map_dim, map_height, map_width)

        return x


class SymNavMapper(nn.Module):
    def __init__(self, map_height, map_width, num_views, img_height, img_width, workspace_size=None, geo_output=False):
        super().__init__()

        assert num_views == 4
        assert img_height == img_width

        self.map_height, self.map_width = map_height, map_width
        self.num_views = num_views
        self.img_height, self.img_width = img_height, img_width
        self.img_rgb = 3
        self.geo_output = geo_output

        self.hidden_dim = [32, 64]
        self.img_embed_dim = 256
        self.map_dim = 1

        # step 1: use regular CNN to embed 4 images
        # input = [batch, map_height, map_width, (4, img_height, img_width, RGB)]
        # output = [batch, map_height, map_width, (4, img_embedding)]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(self.img_rgb, self.hidden_dim[0],
                      kernel_size=10, stride=4, padding=4),
            nn.BatchNorm2d(self.hidden_dim[0]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[0], self.hidden_dim[1],
                      kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(self.hidden_dim[1]),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim[1], self.img_embed_dim,
                      kernel_size=4, stride=1),
            nn.BatchNorm2d(self.img_embed_dim),
            nn.ReLU(),
        )

        # step 2: use E(2)-CNN to process 4 embeddings again
        # Note that dim 1 is group/repr dim, dim -2 & -1 are base space dims
        # input = [batch, 4, img_embedding, (map_height, map_width)]
        # output = [batch, obstacle_embedding, (map_height, map_width)]
        self.r2_space = gspaces.Rot2dOnR2(4)  # TODO a helper to get group
        self.spatial_in_type = FieldType(self.r2_space, self.img_embed_dim * [self.r2_space.regular_repr])
        self.spatial_hid_type = FieldType(self.r2_space, self.hidden_dim[0] * [self.r2_space.regular_repr])
        self.spatial_out_type = FieldType(self.r2_space, 1 * [self.r2_space.trivial_repr])
        self.group_size = self.r2_space.regular_repr.size

        self.map_encoder = e2nn.SequentialModule(
            e2nn.R2Conv(self.spatial_in_type, self.spatial_hid_type,
                        kernel_size=3, stride=1, padding=1),
            e2nn.InnerBatchNorm(self.spatial_hid_type),
            e2nn.ReLU(self.spatial_hid_type),
            e2nn.R2Conv(self.spatial_hid_type, self.spatial_out_type,
                        kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        batch_size, map_height, map_width, num_views, img_height, img_width, img_rgb = x.size()
        assert (
                   map_height, map_width, num_views, img_height, img_width, img_rgb
               ) == (
                   self.map_height, self.map_width, self.num_views, self.img_height, self.img_width, 3
               )

        # > flatten map dimensions, process individual images, shared across batch/locations/views
        x = vis_to_conv2d(x)
        # size = (0) (batch_size * map_width * map_height * |C_4|) x [(1) RGB x (2) image_width x (3) image_height]

        # > step 1
        x = self.img_encoder(x)
        # size = (0) (batch_size * map_width * map_height) x (1) |C_4| x (2) image_embedding

        # > reshape for processing over map, in two steps
        x = x.view(batch_size, num_views, map_height, map_width, self.img_embed_dim)
        x = flatten_repr_channel(x, group_size=self.group_size)

        # > step 2
        x_geo = e2nn.GeometricTensor(x, self.spatial_in_type)
        x_geo = self.map_encoder(x_geo)

        # > return
        if self.geo_output:
            return x_geo
        else:
            x_out = x_geo.tensor
            x_out = x_out.view(batch_size, self.map_dim, map_height, map_width)
            return x_out


class ResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(ResBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(hidden_dim)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),

        )
        self.relu = nn.ReLU(hidden_dim)

        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out


class ResUNet(torch.nn.Module):
    def __init__(self, maze_size, maze_size2, num_views, img_height, img_width, workspace_size=None,
                 n_input_channel=1, n_output_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3):
        super().__init__()
        assert len(n_middle_channels) == 4
        # assert workspace_size % 16 == 0  # Unet requires input is multiplier of 16
        assert workspace_size == 96  # Unet requires input is multiplier of 16
        self.output_size = maze_size

        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]

        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-conv-0', nn.Conv2d(n_input_channel, self.l1_c, kernel_size=3, padding=1, stride=3)),
            ('enc-relu-0', nn.ReLU()),
            ('enc-res-1',
             ResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.MaxPool2d(2)),
            ('enc-res-2',
             ResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.MaxPool2d(2)),
            ('enc-res-3',
             ResBlock(self.l2_c, self.l3_c, kernel_size=kernel_size)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.MaxPool2d(2)),
            ('enc-res-4',
             ResBlock(self.l3_c, self.l4_c, kernel_size=kernel_size)),
        ]))
        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.MaxPool2d(2)),
            ('enc-res-5',
             ResBlock(self.l4_c, self.l4_c, kernel_size=kernel_size)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-res-1',
             ResBlock(2 * self.l4_c, self.l3_c, kernel_size=kernel_size)),
        ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-res-2',
             ResBlock(2 * self.l3_c, self.l2_c, kernel_size=kernel_size)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-res-3',
             ResBlock(2 * self.l2_c, self.l1_c, kernel_size=kernel_size)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-res-4',
             ResBlock(2 * self.l1_c, n_output_channel, kernel_size=kernel_size)),
            ('dec-conv-4', nn.Conv2d(n_output_channel, 1, kernel_size=3, padding=1, padding_mode='circular')),
        ]))

        self.upsample_16_8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_8_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_4_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forwardEncoder(self, obs):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        concat_8 = torch.cat((feature_map_8, self.upsample_16_8(feature_map_16)), dim=1)
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4, self.upsample_8_4(feature_map_up_8)), dim=1)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2, self.upsample_4_2(feature_map_up_4)), dim=1)
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1, self.upsample_2_1(feature_map_up_2)), dim=1)
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        # obs = F.pad(obs, (self.padding,) * 4) if self.padding != 0 else obs
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.forwardEncoder(obs)
        out = self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)
        # out = out[:, :, self.padding:-self.padding, self.padding:-self.padding] if self.padding != 0 else out
        out = F.interpolate(out, (self.output_size,) * 2) if self.output_size != out.shape[-1] else out
        return out
