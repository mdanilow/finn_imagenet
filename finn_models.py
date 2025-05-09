from functools import partial
from typing import Any, Callable, List, Optional

import torch
from torch import nn, Tensor

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity, TruncAvgPool2d
from brevitas.quant import Int32Bias, Int8WeightPerChannelFixedPointMSE, Int8ActPerTensorFixedPoint, Uint8ActPerTensorFixedPoint, Int8BiasPerTensorFixedPointInternalScaling

from brevitas_examples.imagenet_classification.models.common import CommonIntActQuant, CommonUintActQuant
from brevitas_examples.imagenet_classification.models.common import CommonIntWeightPerChannelQuant


class MyIntActQuant(CommonIntActQuant):
    scaling_min_val = 2e-5


class MyUintActQuant(CommonUintActQuant):
    scaling_min_val = 2e-5


# ------------------------- MOBILENETV2

class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 groups=1,
                 norm_layer=None):
        self.padding = (kernel_size - 1) // 2 if padding == None else padding
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, self.padding, groups=groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.activation = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
    

# ------------- QUANT MOBILENETV2

class QuantConvBNReLU(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            weight_bit_width=8,
            act_bit_width=8,
            groups=1,
            bn_eps=1e-5,
            activation_scaling_per_channel=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            act_quant=MyUintActQuant):
        self.padding = (kernel_size - 1) // 2 if padding == None else padding
        layers = []
        layers.append(
            QuantConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                groups=groups,
                bias=False,
                weight_quant=weight_quant,
                weight_bit_width=weight_bit_width
            )
        )
        layers.append(
            nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        )
        layers.append(
            QuantReLU(
                act_quant=act_quant,
                bit_width=act_bit_width,
                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                scaling_stats_permute_dims=(1, 0, 2, 3),
                scaling_per_output_channel=activation_scaling_per_channel,
                return_quant_tensor=True
            )
        )
        super().__init__(*layers)
    

class QuantInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio=1, weight_bit_width=8, act_bit_width=8,
                 skip=True, common_quant=None, dw_act_per_channel=True, requant_sum=False,
                 weight_quant=CommonIntWeightPerChannelQuant, identity_quant=MyIntActQuant, act_quant=MyUintActQuant):
        super(QuantInvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup and skip
        self.requant_sum = requant_sum

        if common_quant is not None:
            self.extra_identity = False
            self.identity = common_quant
        else:
            self.extra_identity = True
            self.identity = QuantIdentity(
                                      act_quant=identity_quant,
                                      bit_width=act_bit_width,
                                      per_channel_broadcastable_shape=(1, oup, 1, 1),
                                      scaling_per_output_channel=False,
                                      return_quant_tensor=True)
        if requant_sum and self.use_res_connect:
            self.sum_quant = QuantIdentity(
                                      act_quant=identity_quant,
                                      bit_width=act_bit_width,
                                      per_channel_broadcastable_shape=(1, oup, 1, 1),
                                      scaling_per_output_channel=False,
                                      return_quant_tensor=True)
        
        layers = []
        # pw
        if expand_ratio != 1:
            layers.append(QuantConvBNReLU(inp, hidden_dim, kernel_size=1,
                                          weight_bit_width=weight_bit_width,
                                          act_bit_width=act_bit_width,
                                          activation_scaling_per_channel=dw_act_per_channel,
                                          weight_quant=weight_quant,
                                          act_quant=act_quant))
        
        layers.extend([
            # dw
            QuantConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, weight_bit_width=weight_bit_width, 
                            act_bit_width=act_bit_width, stride=stride, groups=hidden_dim,
                            weight_quant=weight_quant, act_quant=act_quant),
            
            # pw-linear
            QuantConv2d(
            in_channels=hidden_dim,
            # in_channels=inp,
            out_channels=oup,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            weight_quant=weight_quant,
            weight_bit_width=weight_bit_width),
            
            # bn
            nn.BatchNorm2d(num_features=oup),
            # self.activation
            self.identity
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            if self.extra_identity:
                x = self.identity(x) + self.conv(x)
            else:
                x = x + self.conv(x)
            
            if self.requant_sum:
                x = self.sum_quant(x)
        else:
            x = self.conv(x)

        return x
        

class QuantMobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 weight_bit_width=8,
                 act_bit_width=8,
                 first_layer_weight_bit_width=8,
                 last_layer_weight_bit_width=8,
                 block=None,
                 norm_layer=None,
                 use_common_quant=False,
                 act_per_channel=True,
                 requant_sum=False,
                 fixed_point=False):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(QuantMobileNetV2, self).__init__()

        if use_common_quant:
            print('==> Using common quantizers for skip connections in QuantMobileNetV2')
        if fixed_point:
            print('==> Using fixed point quantization')

        if block is None:
            block = QuantInvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        weight_quant = Int8WeightPerChannelFixedPointMSE if fixed_point else CommonIntWeightPerChannelQuant
        act_quant = Uint8ActPerTensorFixedPoint if fixed_point else MyUintActQuant
        identity_quant = Int8ActPerTensorFixedPoint if fixed_point else MyIntActQuant
        bias_quant = Int8BiasPerTensorFixedPointInternalScaling if fixed_point else Int32Bias

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.input_quant = QuantIdentity(
            act_quant=act_quant,
            bit_width=first_layer_weight_bit_width,
            scaling_per_output_channel=False,
            return_quant_tensor=True
        )
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [QuantConvBNReLU(3, input_channel, stride=2,
                                    weight_bit_width=first_layer_weight_bit_width,
                                    act_bit_width=act_bit_width,
                                    activation_scaling_per_channel=act_per_channel,
                                    weight_quant=weight_quant,
                                    act_quant=act_quant)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                skip_connection_used = (stride == 1) and (input_channel == output_channel)
                features.append(block(input_channel, output_channel, stride=stride, expand_ratio=t,
                                      weight_bit_width=weight_bit_width,
                                      act_bit_width=act_bit_width,
                                      common_quant=common_quant if (skip_connection_used and use_common_quant) else None,
                                      dw_act_per_channel=act_per_channel,
                                      requant_sum=requant_sum,
                                      weight_quant=weight_quant,
                                      act_quant=act_quant,
                                      identity_quant=identity_quant))
                # pass the last quantizer of the block as a common_quant to be used in the next block
                if requant_sum and skip_connection_used:
                    common_quant = features[-1].sum_quant
                else:
                    common_quant = features[-1].identity
                input_channel = output_channel
        # building last several layers
        features.append(QuantConvBNReLU(input_channel, self.last_channel, kernel_size=1, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,
                                        weight_quant=weight_quant, act_quant=act_quant))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.final_pool = TruncAvgPool2d(
            kernel_size=7,
            stride=1,
            bit_width=last_layer_weight_bit_width,
            float_to_int_impl_type='FLOOR')

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            QuantLinear(self.last_channel, num_classes, bias=True, bias_quant=bias_quant, weight_quant=weight_quant,
                        weight_bit_width=last_layer_weight_bit_width),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.input_quant(x)
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = self.final_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)