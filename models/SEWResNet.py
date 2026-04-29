import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from models.layers import TensorNormalization, rate_encode, const_encode

__all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101',
           'sew_resnet152']


class SpikeActIN(nn.Module):
    def __init__(self, T=0, thresh=1.0, gama=1.0):
        super().__init__()
        self.T = T
        self.relu = nn.ReLU(inplace=True)
        # Match the old SEWResNet IF-style dynamics while delegating the spike
        # and surrogate implementations to SpikingJelly.
        self.spike = neuron.IFNode(
            v_threshold=thresh,
            v_reset=0.0,
            surrogate_function=surrogate.PiecewiseQuadratic(alpha=1.0 / gama),
            detach_reset=False,
        )

    def forward(self, x):
        if self.T > 0:
            return self.spike(x)
        return self.relu(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(BasicBlock, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = SpikeActIN()

        self.conv2 = nn.Sequential(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = SpikeActIN()

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(Bottleneck, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Sequential(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = SpikeActIN()

        self.conv2 = nn.Sequential(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = SpikeActIN()

        self.conv3 = nn.Sequential(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = SpikeActIN()

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        out = self.sn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv3[1].bias, 1)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.conv2[1].bias, 1)


class SEWResNet(nn.Module):
    def __init__(self, block, layers, encoding='const', signed=False, atk_encoding='rate', model_encode=False,
                 num_classes=100, zero_init_residual=True, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, T=4, connect_f='ADD'):
        super(SEWResNet, self).__init__()
        self.T = T
        self.encoding = encoding
        self.signed = signed
        self.atk_encoding = atk_encoding
        self.model_encode = model_encode
        self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = SpikeActIN()
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)

        self.mode = 'bptt'
        self.set_simulation_time(T)

    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        self.mode = mode
        functional.set_step_mode(self, 'm' if T > 0 else 's')
        functional.reset_net(self)
        for module in self.modules():
            if isinstance(module, SpikeActIN):
                module.T = T
        return

    def _reshape_to_steps(self, x):
        if self.T <= 0 or x.dim() == 5:
            return x
        if x.dim() != 4:
            raise ValueError(f'expected a 4D flattened time batch or 5D sequence, but got {x.shape}')
        if x.shape[0] % self.T != 0:
            raise ValueError(f'input batch dimension {x.shape[0]} is not divisible by T={self.T}')
        return x.reshape(self.T, x.shape[0] // self.T, *x.shape[1:])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                SpikeActIN()
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if self.model_encode:
            if self.T > 0:
                if self.encoding in ['rate', 'signed', 'hypergeometric']:
                    x = rate_encode(x, self.T, self.encoding)
                elif self.encoding == 'const':
                    x = const_encode(x, self.T)
                else:
                    print("--encoding/atk_encoding not reconginzed")

        x = self._reshape_to_steps(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2) if x.dim() == 5 else torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        functional.reset_net(self)
        try:
            return self._forward_impl(x)
        finally:
            functional.reset_net(self)


def _sew_resnet(block, layers, **kwargs):
    model = SEWResNet(block, layers, **kwargs)
    return model


def sew_resnet18(**kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152(**kwargs):
    return _sew_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    model = sew_resnet34(T=8, connect_f='ADD')
    x = torch.rand(2, 3, 224, 224)
    outputs = model(x).mean(0)
    print(outputs.shape)
    _, predicted = outputs.cpu().max(1)
    print(predicted)
