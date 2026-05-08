from typing import Literal

from torch import nn


class Net(nn.Module):
    def __init__(self, in_dim, p_per_layer, layers, a_fn: Literal["ReLU", "GELU"], out_dim):
        super().__init__()

        activation = nn.GELU if a_fn == "GELU" else nn.ReLU

        self.net = nn.Sequential(
            nn.Linear(in_dim, p_per_layer),
            activation(),
            *[
                nn.Linear(p_per_layer, p_per_layer),
                activation()
            ] * layers,
            nn.Linear(p_per_layer, out_dim),
        )

    def forward(self, x):
        return self.net(x)
