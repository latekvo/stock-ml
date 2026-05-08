from typing import Literal

from torch import nn

Act = Literal["ReLU", "GELU", "SiLU"]

class Net(nn.Module):
    def __init__(self, in_dim, p_per_layer, l_count, act: Act, use_residual: bool, out_dim):
        super().__init__()

        self.use_residual = use_residual

        if act == "GELU":
            activation = nn.GELU
        elif act == "SiLU":
            activation = nn.SiLU
        else:
            activation = nn.ReLU

        layers = [
            nn.Linear(in_dim, p_per_layer),
            activation()
        ]

        for _ in range(l_count):
            layers.append(nn.Linear(p_per_layer, p_per_layer))
            layers.append(activation())

        layers.append(nn.Linear(p_per_layer, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
