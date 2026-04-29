import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate

from models.layers import TensorNormalization, rate_encode, const_encode
from models.spiking_backend import set_default_backend

cfg = {
    'vgg5' : [[64, 'A'], 
              [128, 128, 'A'],
              [],
              [],
              []],
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ],
    'vgg13': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A']
    ],
    'vgg16': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512, 512, 'A']
    ],
    'vgg19': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 256, 'A'],
        [512, 512, 512, 512, 'A'],
        [512, 512, 512, 512, 'A']
    ]
}

class SpikeAct(nn.Module):
    def __init__(self, T, thresh=1.0, gama=1.0):
        super().__init__()
        self.T = T
        self.relu = nn.ReLU(inplace=True)
        # A very large tau keeps the old near-IF accumulator behavior while
        # delegating the spike/reset/surrogate implementation to SpikingJelly.
        self.spike = neuron.LIFNode(
            tau=1e9,
            decay_input=False,
            v_threshold=thresh,
            v_reset=0.0,
            # ATan has CUDA codegen support in SpikingJelly's CuPy backend,
            # while PiecewiseQuadratic does not on the version we use here.
            surrogate_function=surrogate.ATan(alpha=1.0 / gama),
            detach_reset=False,
        )

    def forward(self, x):
        if self.T > 0:
            return self.spike(x)
        return self.relu(x)


class VGG(nn.Module):
    def __init__(self, vgg_name, encoding, T, num_class, norm, init_c=3,encode_in=False):
        super(VGG, self).__init__()
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.T = T
        self.init_channels = init_c
        self.encoding = encoding
        self.encode = encode_in
        self.spike_backend = 'torch'

        if vgg_name == 'vgg11' or vgg_name == 'vgg5':
            self.W = 16 
        else:
            self.W = 1
        
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])
        self.classifier = self._make_classifier(num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

        self.mode = 'bptt'
        self.set_simulation_time(T)

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(layer.AvgPool2d(2))
            else:
                layers.append(layer.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(layer.BatchNorm2d(x))
                layers.append(SpikeAct(self.T))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        classifier = [
            layer.Flatten(),
            layer.Linear(512 * self.W, 4096),
            SpikeAct(self.T),
            layer.Linear(4096, 4096),
            SpikeAct(self.T),
            layer.Linear(4096, num_class),
        ]
        return nn.Sequential(*classifier)

    def _reshape_to_steps(self, x):
        if self.T <= 0 or x.dim() == 5:
            return x
        if x.dim() != 4:
            raise ValueError(f'expected a 4D flattened time batch or 5D sequence, but got {x.shape}')
        if x.shape[0] % self.T != 0:
            raise ValueError(f'input batch dimension {x.shape[0]} is not divisible by T={self.T}')
        return x.reshape(self.T, x.shape[0] // self.T, *x.shape[1:])
    
    #pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        self.mode = mode
        functional.set_step_mode(self, 'm' if T > 0 else 's')
        self.spike_backend = set_default_backend(self, T)
        functional.reset_net(self)
        for module in self.modules():
            if isinstance(module, SpikeAct):
                module.T = T
        return
    def forward(self, inputs):
        functional.reset_net(self)
        try:
            if self.encode:
                if self.T > 0:
                    if self.encoding in ['rate','signed','hypergeometric']:
                        inputs = rate_encode(inputs, self.T, self.encoding) #rate encoding or signed or hyper
                    elif self.encoding == 'const':
                        inputs = const_encode(inputs, self.T)    #constant encoding
                    else:
                        print("--encoding not reconginzed")

            inputs = self._reshape_to_steps(inputs)
            out = self.layer1(inputs)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.classifier(out)
            return out
        finally:
            functional.reset_net(self)
    



if __name__ == '__main__':
    model = VGG('vgg11', 'rate', 'rate', 'normal', 4, 10, None, init_c=3)
    x = torch.rand(64,3,32,32)
    labels = torch.rand(64)
    print(model(x,labels))
