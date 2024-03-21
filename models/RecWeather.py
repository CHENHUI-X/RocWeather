from models.Encoder import WeatherEncoder
from models.Decoder import CentralDecoder, convprojection
import torch.nn as nn
import torch
from collections import OrderedDict


class WeaTenc(WeatherEncoder):
    def __init__(self, **kwargs):
        super(WeaTenc, self).__init__(
            img_size=224,
            patch_size=7,
            in_chans=3,
            embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            mlpdrop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            depths=[2, 2, 6, 2],
            sr_ratios=[8, 4, 2, 1],
            focal_levels=[2, 2, 2, 2],
            focal_windows=[3, 3, 3, 3],
            input_resolution=[64, 32, 16, 8],
        )


class WeaTdec(CentralDecoder):
    def __init__(self, **kwargs):
        super(WeaTdec, self).__init__(
            img_size=224,
            patch_size=7,
            in_chans=3,
            embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            mlpdrop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            depths=[2, 2, 6, 2],
            sr_ratios=[8, 4, 2, 1],
            focal_levels=[2, 2, 2, 2],
            focal_windows=[3, 3, 3, 3],
            input_resolution=[64, 32, 16, 8],
        )


class RecWeather(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(RecWeather, self).__init__()
        self.Tenc = WeaTenc()
        self.Tdec = WeaTdec()
        self.convtail = convprojection()

        if path is not None:
            self.load(path)

    def forward(self, x):
        x1 = self.Tenc(x)
        x2 = self.Tdec(x1)
        clean = self.convtail(x1, x2)
        return clean

    def load(self, path):
        """
        Load checkpoint.
        """
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        model_state_dict_keys = self.state_dict().keys()
        checkpoint_state_dict_noprefix = self.strip_prefix_if_present(
            checkpoint["state_dict"], "module."
        )
        self.load_state_dict(checkpoint_state_dict_noprefix, strict=False)
        del checkpoint
        torch.cuda.empty_cache()

    def strip_prefix_if_present(self, state_dict, prefix):
        keys = sorted(state_dict.keys())
        if not all(key.startswith(prefix) for key in keys):
            return state_dict
        stripped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            stripped_state_dict[key.replace(prefix, "")] = value

        return stripped_state_dict


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 256, 256)
    model = RecWeather()
    outs = model(inputs)
    print("\noutputs  \n", outs.shape)
