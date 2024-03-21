# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import numpy as np
import torch
import torch.nn as nn


def fixed_pos_embedding(x):
    seq_len, dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim) / dim))  # theta
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(0, seq_len,
                     dtype=torch.float), inv_freq).to(x)
    )  # cos m * theta or sin m * theta , shape is ( l , dim ) effect by "i , j -> i j"
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1) # 将 同一个q的 两两紧挨的维度进行交换位置 和 符号
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]  # 传进来的是 d/2 的  m * theta * ξ
    # 因为 2个维度一组, 用的是一样的角度和变换 , 所以 需要先扩充复制一下
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos)) # t * scale 实现了对 m * theta的放缩
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin) # 维度上的相加(靠ξ放缩), 长度维度上靠 n 


class XPOS(nn.Module):
    def __init__(
        self, head_dim, scale_base=512
    ):
        super().__init__()
        self.head_dim = head_dim
        self.scale_base = scale_base

        self.register_buffer(
            # 式子(9) 分子分母同乘 dim (不管head dim ,每个头分开编码,互不影响 ) ,
            #  且 r = 0.4
            "scale", (torch.arange(0, head_dim, 2) + \
                      0.4 * head_dim) / (1.4 * head_dim)
        )  # 2 个 维度 共用一个 theta , 一个 ξ , 这个 scale 就是 ξ
        """
                            -----> torch.arange(0, head_dim, 2) ,  ξ_i
                        |
                        |
                        |
                        v
            torch.arange(min_pos, max_pos, 1) 负责 , ( ξ_i )^n 
            
        """

    def forward(self, x, offset=0, downscale=False):
        length = x.shape[1]  # (b , l , d)
        min_pos = -(length + offset) // 2
        max_pos = length + offset + min_pos
        # 2 个 维度 共用一个 theta , 一个 ξ
        scale = self.scale ** torch.arange(min_pos, max_pos,
                                           1).to(self.scale).div(self.scale_base)[:, None]
        # scale shape : ( l, d // 2)
        sin, cos = fixed_pos_embedding(scale)  # sin, cos  shape  ( l, d // 2)

        if scale.shape[0] > length:
            scale = scale[-length:]
            sin = sin[-length:]
            cos = cos[-length:]

        if downscale:
            scale = 1 / scale

        x = apply_rotary_pos_emb(x, sin, cos, scale)
        return x


if __name__ == '__main__':
    xpos = XPOS(64)
    x = torch.rand((8, 16, 64))
    y = xpos(x)
