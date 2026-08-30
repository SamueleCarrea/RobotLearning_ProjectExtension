"""Probe MLP che prova a predire le masse vere dallo hidden state dell'LSTM.

Volutamente piccolo: se un probe cosi' debole ci riesce, l'informazione era
gia' nello hidden state (stessa idea di Alain & Bengio 2016).
"""

import torch
import torch.nn as nn


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 3):
        # input_dim = hidden_size, oppure 2*hidden_size se si concatena c
        # output_dim = 3, il torso non e' randomizzato
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
