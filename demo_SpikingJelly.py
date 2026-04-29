import torch
from spikingjelly.activation_based import neuron, functional
from spikingjelly.activation_based.model import spiking_vgg

# pick one: spiking_vgg11 / 13 / 16 / 19, with or without _bn
net = spiking_vgg.spiking_vgg11_bn(
    pretrained=False,
    spiking_neuron=neuron.LIFNode,
)

# single-step by default; for multi-step / sequence input:
functional.set_step_mode(net, step_mode='m')

# example sequence input: [T, N, C, H, W]
x = torch.rand(4, 2, 3, 224, 224)
y = net(x)

# always reset state between independent samples / batches
functional.reset_net(net)

print(y.shape)
