import torch.nn as nn
encoder_config = {
    'img_size': 224,
    'patch_size': 7,
    'in_chans': 3,
    'embed_dims': [64, 128, 256, 512],
    'num_heads': [1, 2, 4, 8],
    'mlp_ratios': [4, 4, 4, 4],
    'qkv_bias': False,
    'qk_scale': None,
    'mlpdrop_rate': 0.1,
    'attn_drop_rate': 0.1,
    'drop_path_rate': 0.1,
    'norm_layer': nn.LayerNorm,
    'depths': [2, 2, 6, 3],
    'sr_ratios': [8, 4, 2, 1],
    'focal_levels': [2, 2, 2, 2],
    'focal_windows': [3, 3, 3, 3],
    'input_resolution': [64, 32, 16, 8]
}

decoder_config = {
    'img_size': 224,
    'patch_size': 7,
    'in_chans': 3,
    'embed_dims': [64, 128, 256, 512],
    'num_heads': [1, 2, 4, 8],
    'mlp_ratios': [4, 4, 4, 4],
    'qkv_bias': False,
    'qk_scale': None,
    'mlpdrop_rate': 0.1,
    'attn_drop_rate': 0.1,
    'drop_path_rate': 0.1,
    'norm_layer': nn.LayerNorm,
    'depths': [2, 2, 6, 3],
    'sr_ratios': [8, 4, 2, 1],
    'input_resolution': [64, 32, 16, 8]
}
