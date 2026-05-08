from typing import Literal

from torch import nn
from torch.nn import Module

Act = Literal["ReLU", "GELU", "SiLU"]

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, p_per_layer = 24, l_count = 8, act: Act = "GELU", use_residual: bool = False):
        super().__init__()

        self.use_residual = use_residual

        if act == "GELU":
            activation = nn.GELU
        elif act == "SiLU":
            activation = nn.SiLU
        else:
            activation = nn.ReLU

        layers: list[Module] = [
            nn.Linear(in_dim, p_per_layer),
            activation()
        ]

        for _ in range(l_count):
            layers.append(
                nn.Sequential(
                    nn.Linear(p_per_layer, p_per_layer),
                    activation(),
                )
            )

        layers.append(nn.Linear(p_per_layer, out_dim))

        self.net = nn.Sequential(*layers)
        self.layers = layers

    def forward(self, x):
        if not self.use_residual:
            return self.net(x)

        for layer in self.layers:
            y = layer(x)
            if y.shape == x.shape:
                x = y + x
            else:
                x = y

        return x

