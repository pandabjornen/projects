import torch
import torch.nn as nn
import torch.nn.functional as F

class TriangleMultiplicationOutgoing(nn.Module):
    def __init__(self, c_z: int):
        super().__init__()
        # 1) normalisera pair-features över sista dim (kan också vara InstanceNorm etc)
        self.layernorm_z = nn.LayerNorm(c_z)
        # 2) två lineära projiceringslager för a och b
        self.linear_a = nn.Linear(c_z, c_z)
        self.linear_b = nn.Linear(c_z, c_z)
        # 3) gating-linjär + sigmoid
        self.linear_g = nn.Linear(c_z, c_z)
        # 4) utgående sum-LN + projicering tillbaka till c_z
        self.layernorm_out = nn.LayerNorm(c_z)
        self.linear_out = nn.Linear(c_z, c_z)

    def forward(self, z):
        """
        Args:
            z: tensor av shape [B, L, L, c_z] (batch, residu i, residu j, chan)
        Returns:
            z_update: samma shape [B, L, L, c_z]
        """
        # 1) LayerNorm på input-features z_ij
        z_norm = self.layernorm_z(z)                                            # → [B, L, L, c_z]

        # 2) a_ij = sigmoid(Linear(z_norm)), b_ij = Linear(z_norm)
        a = torch.sigmoid(self.linear_a(z_norm))                                # → [B, L, L, c_z]
        b = self.linear_b(z_norm)                                                # → [B, L, L, c_z]

        # 3) gating g_ij = sigmoid(Linear(z_norm))
        g = torch.sigmoid(self.linear_g(z_norm))                                # → [B, L, L, c_z]

        # 4) sum over k: Σ_k [ a_{ik} * b_{jk} ]  (”outgoing”)
        #    vi får en tensor shape [B, L, L, c_z]
        #    einsum: i k c   j k c  → i j c (sum over k)
        m = torch.einsum('bikc,bjkc->bijc', a, b)                                # → [B, L, L, c_z]

        # 5) LayerNorm på den summerade tensorn
        m_norm = self.layernorm_out(m)                                          # → [B, L, L, c_z]

        # 6) en linjär projektion tillbaka, sen gångra med g_ij (gating)
        z_update = g * self.linear_out(m_norm)                                  # → [B, L, L, c_z]

        return z_update




class TriangleMultiplicationIncoming(nn.Module):
    def __init__(self, c_z: int):
        super().__init__()
        # samma uppsättning lager som för outgoing-versionen
        self.layernorm_z = nn.LayerNorm(c_z)
        self.linear_a = nn.Linear(c_z, c_z)
        self.linear_b = nn.Linear(c_z, c_z)
        self.linear_g = nn.Linear(c_z, c_z)
        self.layernorm_out = nn.LayerNorm(c_z)
        self.linear_out = nn.Linear(c_z, c_z)

    def forward(self, z):
        """
        Samma som outgoing, men summation över 'incoming'-trianglar:
        Σ_k [ a_{ki} * b_{kj} ], dvs vänd ordningen på index i↔k
        """
        z_norm = self.layernorm_z(z)                                            # [B, L, L, c_z]
        a = torch.sigmoid(self.linear_a(z_norm))                                # [B, L, L, c_z]
        b = self.linear_b(z_norm)                                                # [B, L, L, c_z]
        g = torch.sigmoid(self.linear_g(z_norm))                                # [B, L, L, c_z]

        # nu Σ_k a_{k,i} * b_{k,j}
        # vi byter plats på i och k i a: (b,k,i,c) men i vår storage är alltid första index residu i
        # så vi kastar om dim för att komma åt rätt kombination
        # enklast med einsum:
        m = torch.einsum('bkic,bkjc->bijc', a, b)                                # [B, L, L, c_z]

        m_norm = self.layernorm_out(m)                                          # [B, L, L, c_z]
        z_update = g * self.linear_out(m_norm)                                  # [B, L, L, c_z]
        return z_update
