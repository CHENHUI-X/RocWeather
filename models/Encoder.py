from models.Blocks import OverlapPatchEmbed, RPLFABlock, FocalNetBlock
import torch.nn as nn
from models.Xpos import XPOS
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class WeatherEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=7, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, mlpdrop_rate=0.1,
                 attn_drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm, depths=[2, 2, 6, 3],
                 sr_ratios=[8, 4, 2, 1],focal_levels=[2, 2, 2, 2],focal_windows=[3, 3, 3, 3],input_resolution=[64, 32, 16, 8]):
        '''

        :param img_size:
        :param patch_size:
        :param in_chans:
        :param embed_dims:  embed_dims = [64, 128, 320, 512] 外边传进来的参数
        :param num_heads:
        :param mlp_ratios:
        :param qkv_bias:
        :param qk_scale:
        :param mlpdrop_rate:
        :param attn_drop_rate:
        :param drop_path_rate:
        :param norm_layer:
        :param depths:   TransformerSubBlock layer num in a BLock , that many Block form the Transformer Encoder
        :param sr_ratios:  attention key scaler factor
        :param block_num: 4
        :param window_size:  8
        :param input_resolution: [64,32,16,8]
        '''
        super().__init__()
        self.embed_dims = embed_dims
        self.input_resolution = input_resolution
        self.depths = depths
        self.pe = XPOS(embed_dims[0])
        # active function
        self.active = nn.GELU()
        self.path_embed = nn.ModuleList(
            [
                OverlapPatchEmbed(
                    input_size=img_size, patch_size=patch_size, stride=4, in_chans=in_chans, embed_dim=embed_dims[0]),
                OverlapPatchEmbed(
                    input_size=img_size // 4, patch_size=patch_size, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]),
                OverlapPatchEmbed(
                    input_size=img_size // 8, patch_size=patch_size, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]),
                OverlapPatchEmbed(
                    input_size=img_size // 16, patch_size=patch_size, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
            ]
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.RPLFA = nn.ModuleList()  # 基于相对位置的低频感知模块
        self.norm = nn.ModuleList()  # nrom layers
        self.HFDC = nn.ModuleList()  # 尺度高频感知模块
        for i in range(4):
            self.RPLFA.append(
                nn.ModuleList(
                    [
                        RPLFABlock(
                            dim=embed_dims[i],
                            num_heads=num_heads[i],
                            mlp_ratio=mlp_ratios[i],
                            mlp_drop=mlpdrop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[cur + j],
                            norm_layer=norm_layer,
                        ) for j in range(depths[i])
                    ]
                )
            )
            self.norm.append(norm_layer(embed_dims[i]))
            self.HFDC.append(
                nn.ModuleList(
                    [
                        FocalNetBlock(
                            dim=embed_dims[i],
                            input_resolution=to_2tuple(input_resolution[i]),
                            mlp_ratio=mlp_ratios[i],
                            drop=mlpdrop_rate,
                            drop_path=dpr[cur + j],
                            norm_layer=norm_layer,
                            focal_level=focal_levels[i],
                            focal_window=focal_windows[i],
                        ) for j in range(depths[i])
                    ]
                )
            )
            cur += depths[i]
            
        

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.RPLFA[0][i].drop_path.drop_prob = dpr[cur + i]
            self.HFDC[0][i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.RPLFA[1][i].drop_path.drop_prob = dpr[cur + i]
            self.HFDC[1][i].drop_path.drop_prob = dpr[cur + i]


        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.RPLFA[2][i].drop_path.drop_prob = dpr[cur + i]
            self.HFDC[2][i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.RPLFA[3][i].drop_path.drop_prob = dpr[cur + i]
            self.HFDC[3][i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B,C,H,W = x.shape
        num_layers = len(self.depths)
        res = x  # B,C,H,W  
        embed_x = self.path_embed[0](x) 
        inputs = embed_x + self.pe(embed_x) # add position encoding
        features = []
        for i in range(num_layers):
            if i < 1:
                lf_output_pre = inputs
                hf_output_pre = inputs
            else:
                out_pre = self.path_embed[i](
                    out_pre.permute(0, 2, 1).reshape(
                        B, self.embed_dims[i-1], 
                        self.input_resolution[i-1], self.input_resolution[i-1]
                    )
                )
                lf_output_pre = out_pre
                hf_output_pre = out_pre
            for j in range(self.depths[i]):
                lf_output = lf_output_pre + self.RPLFA[i][j](
                    lf_output_pre, self.input_resolution[i], self.input_resolution[i]
                )
                lf_output_pre = lf_output

                hf_output = hf_output_pre + self.HFDC[i][j](
                    hf_output_pre, self.input_resolution[i], self.input_resolution[i]
                )
                hf_output_pre = hf_output
            out_pre = lf_output_pre + hf_output_pre

            features.append(out_pre)
            
        return features

    def forward(self, x):
        x = self.forward_features(x)
        return x


    