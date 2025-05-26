import torch


class CustomLogger:
    def __init__(self):
        self.content = {}
        self.hist = {}
        self.plot_3d = {}

    def add(self, group, name, value):
        if value is None:
            return

        desc = f"{group}/{name}"

        if isinstance(value, torch.Tensor):
            value = value.mean().detach().cpu()

        if desc in self.content.keys():
            self.content[desc] = value * 0.01 + self.content[desc] * 0.99
        else:
            self.content[desc] = value

    def reset(self):
        self.content = {}

    def add_tensor_stats(self, group, name, tensor):
        self.add(group, name + " min()", tensor.min())
        self.add(group, name + " mean()", tensor.mean())
        self.add(group, name + " std()", tensor.std())
        self.add(group, name + " max()", tensor.max())
