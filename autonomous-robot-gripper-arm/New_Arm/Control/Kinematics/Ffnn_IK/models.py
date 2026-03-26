import torch.nn as nn
import torch


class IKNet_arm(nn.Module):
    def __init__(self, hidden_dim=124):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, points):
        raw_angles = self.net(points)

        # small buffer allows smooth gradients near boundaries
        eps = 0.1 

        def scale_tanh(x, min_val, max_val):
            return ((torch.tanh(x) + 1) / 2) * (max_val - min_val) + min_val

        a = scale_tanh(raw_angles[:, 0], -torch.pi * (1 + eps), torch.pi * (1 + eps))
        b = scale_tanh(raw_angles[:, 1], -torch.pi/3 * (1 + eps), torch.pi/3 * (1 + eps))
        c = scale_tanh(raw_angles[:, 2], -torch.pi/2 * (1 + eps), torch.pi/3 * (1 + eps))
        d = scale_tanh(raw_angles[:, 3], -torch.pi/2 * (1 + eps), 0.0 + eps)

        angles = torch.stack([a, b, c, d], dim=1)
        return angles



class IKNet(nn.Module):
    def __init__(self, hidden_dim=156):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, points):
        raw_angles = self.net(points)

        # small buffer allows smooth gradients near boundaries
        eps = 0.1 

        def scale_tanh(x, min_val, max_val):
            return ((torch.tanh(x) + 1) / 2) * (max_val - min_val) + min_val

        a = scale_tanh(raw_angles[:, 0], -torch.pi * (1 + eps), torch.pi * (1 + eps))
        b = scale_tanh(raw_angles[:, 1], -torch.pi/3 * (1 + eps), torch.pi/3 * (1 + eps))
        c = scale_tanh(raw_angles[:, 2], -torch.pi/2 * (1 + eps), torch.pi/3 * (1 + eps))
        d = scale_tanh(raw_angles[:, 3], -torch.pi/2 * (1 + eps), 0.0 + eps)

        angles = torch.stack([a, b, c, d], dim=1)
        return angles
