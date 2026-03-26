import torch
import torch.nn as nn
from torch.optim import Adam

from model.architecture.encoderOnlyTransformer.AttentionSelf import Attention
from torch.nn import MultiheadAttention

from model.architecture.encoderOnlyTransformer.PositionalEncodingg import PositionEncoding
import lightning as L 

import time

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nrOfHeads, dropoutRate, max_context_size):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nrOfHeads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropoutRate)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.max_context_size = max_context_size

    def forward(self, x):
        # Get sequence length dynamically from input x
        seq_len = x.size(1)
        # Pass the dynamic seq_len to get_attention_mask
        attn_out, _ = self.attn(x, x, x, attn_mask=self.get_attention_mask(seq_len))
        x = self.norm1(x + self.dropout(attn_out))
        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout(mlp_out))
        return x

    def get_attention_mask(self, seq_len):
        if self.max_context_size is None:
            return None  # No masking if no window is defined

        # Create a mask suitable for MultiheadAttention (float type, 0 for attend, -inf for mask)
        # Or a boolean mask (True for masked positions) - let's use boolean
        mask = torch.ones((seq_len, seq_len), device="mps", dtype=torch.bool) # Use input tensor's device

        for i in range(seq_len):
            start = max(0, i - self.max_context_size)
            end = min(seq_len, i + self.max_context_size + 1)
            # Set positions *within* the context window to False (don't mask)
            mask[i, start:end] = False

        # Return the 2D mask directly. MultiheadAttention will broadcast it.
        # Expected format for boolean mask: True indicates a masked position.
        return mask

class EncoderOnlyTransformer(L.LightningModule): 

    def __init__(self, maxLengthProtein, d_model, printEveryEpoch, nrOfHeads, dropoutRate, learningRate, nrOfTEncoderLayers, max_context_size):
        super().__init__()

        self.train_rmse_log = []
        self.val_rmse_log = []
        self.epoch_times = []

        self.learningRate = learningRate
        self.print_every = printEveryEpoch

        self.we = nn.Embedding(num_embeddings = maxLengthProtein, embedding_dim=d_model) #word embedding
        
        self.pe = PositionEncoding(d_model, maxLengthProtein)

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(d_model, nrOfHeads, dropoutRate, max_context_size)
            for _ in range(nrOfTEncoderLayers) 
        ])

        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropoutRate)

        self.loss = nn.MSELoss(reduction='none') 

        self.to("mps")

    def forward(self, token_ids):
        x = self.pe(self.we(token_ids))
        x = self.encoder(x)

        batch_size, seq_len, _ = x.size()

        # Iterera genom batchen för att beräkna relationerna mer minnesvänligt
        pairwise = []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:  # Undvik diagonalen
                    pairwise.append(torch.cat([x[:, i, :], x[:, j, :]], dim=-1))
        
        # Kombinera pairwise-listan till en tensor (detta ger en mycket mer minnesvänlig beräkning)
        pairwise = torch.stack(pairwise, dim=1)
        pairwise = pairwise.view(batch_size, seq_len, seq_len, -1)  # Forma om till (batch_size, seq_len, seq_len, 2*d_model)

        out = self.pair_mlp(pairwise).squeeze(-1)
        out = 0.5 * (out + out.transpose(1, 2))  # Symmetriska distansmatriser
        return out


    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.learningRate)
    
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch 
        output = self.forward(input_tokens)
        
        # Maskera diagonalen
        mask = torch.eye(output.shape[1], device=output.device)  # Skapar en mask för diagonalen
        loss = self.loss(output, labels)  # Beräkna förlust per element
        
        # Maskera diagonalen (ignorera diagonalen vid förlustberäkningen)
        loss = loss * (1 - mask)  # Nollställ diagonalen
        loss = loss.sum() / (loss != 0).sum()  # Normalisera så att förlusten inte påverkas av diagonalen

        rmse = torch.sqrt(loss)
        self.log("train_rmse", rmse, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tokens, labels = batch 
        output = self.forward(input_tokens)
        
        # Maskera diagonalen
        mask = torch.eye(output.shape[1], device=output.device)
        loss = self.loss(output, labels)  # Beräkna förlust per element
        
        # Maskera diagonalen (ignorera diagonalen vid förlustberäkningen)
        loss = loss * (1 - mask)  # Nollställ diagonalen
        loss = loss.sum() / (loss != 0).sum()  # Normalisera så att förlusten inte påverkas av diagonalen

        rmse = torch.sqrt(loss)
        self.log("val_rmse", rmse, prog_bar=False)
        return loss

    def on_train_epoch_start(self):

        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        train_rmse = self.trainer.callback_metrics.get("train_rmse")
        val_rmse = self.trainer.callback_metrics.get("val_rmse")

        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        self.train_rmse_log.append(train_rmse.item())
        self.val_rmse_log.append(val_rmse.item())

        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.trainer.max_epochs - self.current_epoch - 1
        eta = avg_time * remaining_epochs

        if self.current_epoch % self.print_every == 0:
            print(f"Epoch {self.current_epoch}: RMSE train: {train_rmse:.4f} Å , val: {val_rmse:.4f} Å")
            print(f"ETA: {eta/60:.1f} min")
        
