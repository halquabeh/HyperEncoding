"""Minimal attack base (replaces torchattacks.attack.Attack for imports).

Importing ``torchattacks`` loads the whole package (including SciPy), which can
fail on systems where conda's SciPy needs a newer ``libstdc++`` than the
default system linker provides. These attacks only need the behavior below.
"""
from collections import OrderedDict

import torch


class Attack:
    def __init__(self, name, model):
        self.attack = name
        self._attacks = OrderedDict()
        self.set_model(model)
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = None

        self.attack_mode = "default"
        self.supported_mode = ["default"]
        self.targeted = False
        self._target_map_function = None

        self.normalization_used = None
        self._normalization_applied = None

        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def set_device(self, device):
        self.device = device

    def set_model_training_mode(
        self, model_training=False, batchnorm_training=False, dropout_training=False
    ):
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    def _change_model_mode(self, given_training):
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training and "BatchNorm" in m.__class__.__name__:
                    m.eval()
                if not self._dropout_training and "Dropout" in m.__class__.__name__:
                    m.eval()
        else:
            self.model.eval()

    def _recover_model_mode(self, given_training):
        if given_training:
            self.model.train()

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied is False:
            inputs = self.normalize(inputs)
        return self.model(inputs)

    def get_target_label(self, inputs, labels=None):
        if self._target_map_function is None:
            raise ValueError(
                "target_map_function is not initialized by set_mode_targeted."
            )
        if self.attack_mode == "targeted(label)":
            target_labels = labels
        else:
            target_labels = self._target_map_function(inputs, labels)
        return target_labels

    def _get_target_label(self, inputs, labels=None):
        return self.get_target_label(inputs, labels)

    def __call__(self, inputs, labels=None, *args, **kwargs):
        given_training = self.model.training
        self._change_model_mode(given_training)

        if self._normalization_applied is True:
            inputs = self.inverse_normalize(inputs)
            self._set_normalization_applied(False)
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)
            adv_inputs = self.normalize(adv_inputs)
            self._set_normalization_applied(True)
        else:
            adv_inputs = self.forward(inputs, labels, *args, **kwargs)

        self._recover_model_mode(given_training)
        return adv_inputs

    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    def normalize(self, inputs):
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return inputs * std + mean

    def forward(self, inputs, labels=None, *args, **kwargs):
        raise NotImplementedError
