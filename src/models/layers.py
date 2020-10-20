import torch
import torch.nn as nn


class ControlActivationLayer(nn.Module):

    def forward(self, x):
        # split into thrust and other commands and perform activation
        thrust_tensor = torch.sigmoid(x[:, 0:1])
        other_tensor = torch.tanh(x[:, 1:])

        # recombine and return result
        x = torch.cat((thrust_tensor, other_tensor), dim=1)
        return x
