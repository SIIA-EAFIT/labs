from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

class ResidualBlock(nn.Module):
    """
    Resnet building block

    y = F(x, {Wi}) + x.
    """
    def __init__(
        self, 
        in_channels: int,
        building_params: List[Tuple[int, int]], 
        downsampling: bool = False,
    ):
        """
        Parameters
        ----------
        building_params
            Example:
                [
                    (3, 64), # (3 x 3), 64
                    (3, 64), # (3 x 3), 64
                ]
        """
        super().__init__()

        assert len(building_params) > 1, ValueError

        self.downsampling = downsampling
        self.activation = nn.ReLU

        self.net = self.make_block(building_params)
        self._last_activation = self.activation()

        _, out_channels = building_params[-1] 

        if in_channels != out_channels or downsampling:
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2 if downsampling else 1
            )
        else:
            self.proj = nn.Identity()
    
    def _conv_bn(self, in_channels: int, out_channels: int, kernel_size: int) -> List[nn.Module]:
        return [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                bias=False, # Batchnorm handle the bias!!!
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2, # (W - F + 2P) / S + 1 == W 
            ),
            nn.BatchNorm2d(out_channels)
        ]

    def make_block(self, params: List[Tuple[int, int]]) -> nn.Module:
        blocks = []
        for prev, curr in zip(params, params[1:]):
            in_kernel_size, in_channels  = prev
            out_kernel_size, out_channels  = curr

            blocks.extend(
                self._conv_bn(in_channels, out_channels, in_kernel_size)
            )
            blocks.append(self.activation())

        blocks.extend(
            self._conv_bn(out_channels, out_channels, out_kernel_size)
        )

        return nn.Sequential(*blocks)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self._last_activation(self.net(input_tensor) + self.proj(input_tensor))
        
        return output

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_weights()

    def init_weights(self):
        for name, module in self.named_modules():
            pass # init modules

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass