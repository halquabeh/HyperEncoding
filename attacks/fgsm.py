import torch
import torch.nn as nn


class FGSM:
    r"""
    altered from torchattack
    """
    def __init__(self, model, device, forward_function=None, eps=0.007, T=None, signed=False, m=1, **kwargs):
        # super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.forward_function = forward_function
        self.T = T
        self.avg_k = 0
        self.counter = 0
        self.signed = signed
        self.m=m
        self.model = model
        self.device = device
    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        if self.forward_function is not None:
            outputs = self.forward_function(self.model, images, self.T)
            for i in range(self.m-1):
                outputs += self.forward_function(self.model, images, self.T)
        else:
            outputs = self.model(images)    

        cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=-1*self.signed, max=1).detach()

        return adv_images
        # Make the class callable so it can be used as atk(inputs, labels)
    def __call__(self, images, labels):
        return self.forward(images, labels)
 
