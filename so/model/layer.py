#!/usr/bin/env python

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from dgl.nn.pytorch import GATConv, EdgeConv, GATv2Conv, SAGEConv, GraphConv


# class ZScoreDeZScore(nn.Module):
#     def __init__(self, mu_bg, std_bg):
#         super().__init__()
#         self.register_buffer("mu_bg", torch.from_numpy(mu_bg))
#         self.register_buffer("std_bg", torch.from_numpy(std_bg))
#         print('ZScoreDeZScore layer')
#     def zscore(self, x, bidx):
#         mu, std = self.mu_bg[bidx], self.std_bg[bidx]
#         return (x - mu) / std
#
#     def dezscore(self, z, bidx):
#         mu, std = self.mu_bg[bidx], self.std_bg[bidx]
#         return z * std + mu


class ZScoreDeZScore(nn.Module):
    def __init__(self, mu_bg, std_bg, eps=1e-6, clip_value=10.0):
        super().__init__()
        self.register_buffer("mu_bg", torch.from_numpy(mu_bg))
        self.register_buffer("std_bg", torch.from_numpy(std_bg))
        self.eps = eps
        self.clip_value = clip_value
        print('clip')

    def zscore(self, x, bidx):
        mu = self.mu_bg[bidx]
        std = self.std_bg[bidx]
        std_safe = torch.clamp(std, min=self.eps)
        z = (x - mu) / std_safe
        z = torch.clamp(z, -self.clip_value, self.clip_value)
        return z

    def dezscore(self, z, bidx):
        mu = self.mu_bg[bidx]
        std = self.std_bg[bidx]
        std_safe = torch.clamp(std, min=self.eps)
        return z * std_safe + mu


def reparameterize(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, input_dim, h_dim=16, multiple=8, n_domain=None, num_heads=2):
        """

        :param input_dim:
        :param h_dim:
        :param multiple:
        :param num_heads:
        """
        super().__init__()
        input_dim = input_dim
        input_dim2 = h_dim * multiple
        h_dim = h_dim
        # encode
        # self.conv1 = GATv2Conv(input_dim, input_dim2, num_heads=1, bias=True, activation=None)
        # self.conv2 = GATv2Conv(input_dim2, input_dim2, num_heads=num_heads, bias=True, activation=None)
        self.conv1 = GraphConv(input_dim, input_dim2)
        self.conv2 = GraphConv(input_dim2, input_dim2, )

        # reparameterize
        # self.conv_mean = GATv2Conv(input_dim2, h_dim, num_heads=num_heads, bias=True, activation=None)
        # self.conv_var = GATv2Conv(input_dim2, h_dim, num_heads=num_heads, bias=True, activation=None)
        self.conv_mean = GraphConv(input_dim2, h_dim, )
        self.conv_var = GraphConv(input_dim2, h_dim, )

        # reset_parameters
        self.reset_parameters()

    def reset_parameters(self):
        # self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        # self.conv_mean.reset_parameters()
        # self.conv_var.reset_parameters()
        pass

    def forward(self, x, block=None):
        """
        """
        o = self.conv1(block, x)
        o = self.conv2(block, o) + o
        mean = self.conv_mean(block, o)
        var = torch.exp(torch.clamp(self.conv_var(block, o), min=-10, max=10))
        # mean = self.conv_mean(block, o)
        # var = torch.exp(torch.clamp(self.conv_var(block, o), min=-10, max=10))
        z = reparameterize(mean, var)
        return z, mean, var


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self, h_dim, input_dim, multiple=8, n_domain=None, num_heads=2):
        """

        :param h_dim:
        :param input_dim:
        :param multiple:
        :param n_domain:
        """
        super().__init__()
        input_dim2 = h_dim * multiple
        h_dim = h_dim
        # encode
        # self.conv1 = GATv2Conv(h_dim+h_dim, input_dim2, num_heads=num_heads, bias=True, activation=None)
        self.conv1 = GraphConv(h_dim+h_dim, input_dim2, )
        self.norm1 = nn.BatchNorm1d(input_dim2)
        self.act1 = nn.LeakyReLU()
        # self.conv2 = GATv2Conv(input_dim2, input_dim, num_heads=1, bias=True, activation=None)
        self.conv2 = GraphConv(input_dim2, input_dim, )
        self.act2 = nn.ReLU()
        self.act3 = nn.Sigmoid()
        self.batch_embedding = nn.Embedding(n_domain, h_dim)
        # reset_parameters
        self.reset_parameters()

    def reset_parameters(self):
        # self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        self.norm1.reset_parameters()

    def forward(self, x, y=None, block=None):
        batch_o = self.batch_embedding(y)
        x = torch.hstack([x, batch_o])
        o = self.conv1(block, x)
        o = self.act1(self.norm1(o))  # LeakyReLU
        o = self.conv2(block, o)
        o_sig = self.act3(o)
        return o, o_sig
