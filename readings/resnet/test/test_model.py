import torch
from torch import nn
from torch import testing
from torchvision.models import resnet18
from src.model import ResidualBlock

def test_basic_residual_block():
    tv_model = resnet18()
    our_layer1 = ResidualBlock(64, [(3, 64), (3, 64)])
    tv_layer1 = tv_model.layer1[0]

    layer_iter = zip(tv_layer1.modules(),  our_layer1.net.modules())
    next(layer_iter)

    for tv, our in layer_iter:
        assert type(tv) == type(our)

        if isinstance(tv, nn.Conv2d) or isinstance(tv, nn.BatchNorm2d):
            assert our.weight.shape == tv.weight.shape

            our.weight = tv.weight # tie weights for testing purposes
    
    input_tensor = torch.randn(1, 64, 8, 8)
    our_output = our_layer1(input_tensor)
    tv_output = tv_layer1(input_tensor)

    testing.assert_close(our_output, tv_output, rtol=0, atol=0)

# TODO: test bottleneck implementation