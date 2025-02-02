import numpy as np
import torch.nn as nn
from helper import valid


class APoZ:
    def __init__(self, model):
        self.set_model_and_cfgs(model)
        self.idx = 0
        self.hook_handles = []
        print(f"Layer(ReLU + Linear) {self.num_layer} module register")

    # get the # filters in features and classifiers.
    def set_model_and_cfgs(self, model):
        self.num_layer = 0
        self.apoz = []
        self.model = model
        cfg = []
        cfg_names = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d)):
                cfg.append(layer.out_channels)
                cfg_names.append(f"{name}_conv")

            if isinstance(layer, (nn.Linear)):
                cfg.append(layer.out_features)
                cfg_names.append(f"{name}_linear")

            if isinstance(layer, (nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                cfg.append('M')

        # delete the last one since we don't trim it.
        del cfg[-1]
        del cfg_names[-1]
        self.cfgs = cfg
        self.layer_names = cfg_names
        # M is for Pooling layers.
        for c in self.cfgs:
            if c == 'M':
                continue
            self.apoz.append([0] * c)
            self.num_layer += 1

        self.apoz = np.array(self.apoz, dtype=object)

    def get_zero_percent_hook(self, module, input, output):
        if output.dim() == 4:
            p_zero = (output == 0).sum(dim=(2, 3)).float() / (output.size(2) * output.size(3))
            self.apoz[self.idx] += p_zero.mean(dim=0).cpu().numpy()
        elif output.dim() == 2:
            # 2D for linear layers
            p_zero = (output == 0).sum(dim=0).float() / output.size(0)
            self.apoz[self.idx] += p_zero.cpu().numpy()
        else:
            raise ValueError(f"{output.dim()} dimension is Not Supported")

        self.idx += 1

        if self.idx == self.num_layer:
            self.idx = 0

    # APoZ run after the output from the ReLU. ie in a forward hook
    def register(self):
        # register function as forward hook for relu layer
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.ReLU)):
                self.hook_handles.append(layer.register_forward_hook(self.get_zero_percent_hook))

        # for module in self.model.features.modules():
        #     if type(module) == nn.ReLU:
        #         self.hook_handles.append(module.register_forward_hook(self.get_zero_percent_hook))
        #
        # for module in self.model.classifier.modules():
        #     if type(module) == nn.ReLU:
        #         self.hook_handles.append(module.register_forward_hook(self.get_zero_percent_hook))

    def get_apoz(self, loader, criterion, device='cpu'):
        top1, top5 = valid(self.model,
                           loader,
                           criterion,
                           device)

        print(f"top1 : {top1} top5 : {top5}")

        # normalise . The N from the formula
        return self.apoz / len(loader)

    def deregister(self):
        [handle.remove() for handle in self.hook_handles]
