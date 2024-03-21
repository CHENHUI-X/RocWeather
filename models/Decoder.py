import torch.nn as nn
from models.Blocks import OverlapPatchEmbed,Block_dec,UpsampleConvLayer,ResidualBlock,ConvLayer
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
class CentralDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=7, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, mlpdrop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=[2, 2, 6, 3],
                 sr_ratios=[8, 4, 2, 1],focal_levels=[2, 2, 2, 2],focal_windows=[3, 3, 3, 3],input_resolution=[64, 32, 16, 8]):
        
        super().__init__()
        self.depths = depths
        self.input_resolution = input_resolution
        self.H = self.W = self.input_resolution[-1]
        # transformer decoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,sum(depths))]  # stochastic depth decay rule
        cur = sum(depths[:-1])
        self.block = nn.ModuleList(
            [
                Block_dec(
                    dim=embed_dims[-1],
                    num_heads=num_heads[-1],
                    mlp_ratio=mlp_ratios[-1],
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    mlp_drop=mlpdrop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[-1])
                for i in range(depths[-1])
            ]
            )
        self.norm = norm_layer(embed_dims[-1])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = sum(self.depths[:-1])
        for i in range(self.depths[-1]):
            self.block[i].drop_path.drop_prob = dpr[cur + i]


    def forward_features(self, features):
        res = x = features[-1]
        B = x.shape[0]

        # stage 1
        for i, blk in enumerate(self.block):
            x = blk(x, self.H, self.W)
        x = self.norm(x)
        # x = x.reshape(B,self.H, self.W, -1).permute(0, 3, 1, 2).contiguous()
        return res + x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class convprojection(nn.Module):
    def __init__(self, input_resolution=[64, 32, 16, 8], **kwargs):
        super(convprojection, self).__init__()
        self.input_resolution = input_resolution
        self.dense_6 = nn.Sequential(ResidualBlock(512))
        self.convd32x = UpsampleConvLayer(512, 256, kernel_size=4, stride=2) # 16 * 16 = 256
        self.dense_5 = nn.Sequential(ResidualBlock(256))
        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=4, stride=2)# 32 * 32 = 1024
        self.dense_4 = nn.Sequential(ResidualBlock(128))
        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2) # 64 * 64  = 4096
        self.dense_3 = nn.Sequential(ResidualBlock(64))
        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=4, stride=2) # 128 * 128
        self.dense_2 = nn.Sequential(ResidualBlock(32))
        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=4, stride=2) # 256 * 256
        self.dense_1 = nn.Sequential(ResidualBlock(16))
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()

    def forward(self, encoder_outs, decoder_outs):
        B, _, _ = decoder_outs.shape
        fe1, fe2, fe3, fe4 = [
            out.reshape(
                B, self.input_resolution[i], self.input_resolution[i], -1
            ).permute(0, 3, 1, 2).contiguous()
            for i, out in enumerate(encoder_outs)
        ]
        fd = decoder_outs.reshape(
                B, self.input_resolution[-1], self.input_resolution[-1], -1
            ).permute(0, 3, 1, 2).contiguous()
        
        res32x = self.dense_6(fd) + fe4
        
        res16x = self.convd32x(res32x) 
        res16x = self.dense_5(res16x) + fe3

        res8x = self.convd16x(res16x)
        res8x = self.dense_4(res8x) + fe2
        
        res4x = self.convd8x(res8x)
        res4x = self.dense_3(res4x) + fe1
        
        res2x = self.convd4x(res4x)
        res2x = self.dense_2(res2x)
        x = self.convd2x(res2x)
        x = self.dense_1(x)
        x = self.active(self.conv_output(x))
        return x

