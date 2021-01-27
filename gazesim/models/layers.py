import torch
import torch.nn as nn


class LoadableModule(nn.Module):

    def load_model_info(self, model_info_dict):
        self.load_state_dict(model_info_dict["model_state_dict"])


class ControlActivationLayer(nn.Module):

    def forward(self, x):
        # split into thrust and other commands and perform activation
        thrust_tensor = torch.sigmoid(x[:, 0:1])
        other_tensor = torch.tanh(x[:, 1:])

        # recombine and return result
        x = torch.cat((thrust_tensor, other_tensor), dim=1)
        return x


class DummyLayer(nn.Module):

    def forward(self, x):
        return x
