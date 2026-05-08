from torch import nn


class Net(nn.Module):
    def __init__(self, in_dim, p_per_layer, layers, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, p_per_layer),
            nn.ReLU(),
            *[
                nn.Linear(p_per_layer, p_per_layer),
                nn.ReLU()
            ] * layers,
            nn.Linear(p_per_layer, out_dim),
        )

    def forward(self, x):
        return self.net(x)
