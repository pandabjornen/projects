import torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nrOfHeads, dropoutRate):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nrOfHeads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropoutRate)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_out))
        return x