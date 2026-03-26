import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from distance_matrix.model.architecture.encoderOnlyTransformer.PositionalEncodingg import PositionEncoding
from distance_matrix.model.architecture.encoderOnlyTransformer.TransformerEncoderBlock import TransformerEncoderBlock
from distance_matrix.model.architecture.encoderOnlyTransformer.TriangleUpdate import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming
)
import time

class BinsEncoderOnlyTransformer(L.LightningModule):
    def __init__(
        self,
        maxLengthProtein,
        d_model,
        printEveryEpoch,
        nrOfHeads,
        dropoutRate,
        learningRate,
        nrOfTEncoderLayers,
        d_z, nr_bins
    ):
        super().__init__()
        self.L = maxLengthProtein
        self.d_model = d_model
        self.d_z = d_z
        self.nrOfHeads = nrOfHeads
        self.nr_bins = nr_bins # ADDED: Store nr_bins

        self.tri_out = TriangleMultiplicationOutgoing(d_z)
        self.tri_in = TriangleMultiplicationIncoming(d_z)

        self.row_attn = nn.MultiheadAttention(d_z, nrOfHeads, dropout=dropoutRate)
        self.col_attn = nn.MultiheadAttention(d_z, nrOfHeads, dropout=dropoutRate)

        self.seq_embed = nn.Embedding(maxLengthProtein, d_model)
        self.msa_embed = nn.Embedding(maxLengthProtein, d_z)
        self.pe_seq = PositionEncoding(d_model, maxLengthProtein)
        self.pe_msa = PositionEncoding(d_z, maxLengthProtein)

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(d_model, nrOfHeads, dropoutRate)
            for _ in range(nrOfTEncoderLayers)
        ])

        self.layer_norm = nn.LayerNorm(d_z)
        self.transition_a = nn.Linear(d_z, d_z)
        self.transition_b = nn.Linear(d_z, d_z)
        self.outer_linear = nn.Linear(d_z, d_z)

        self.mlp = nn.Sequential(nn.Linear(d_model, d_z), nn.Softplus())
        
        self.pair_mlp = nn.Linear(d_z, self.nr_bins) 

        
        self.loss = nn.CrossEntropyLoss(reduction='mean') 
        self.learningRate = learningRate
        self.print_every = printEveryEpoch
        
        self.train_acc_log = [] 
        self.val_acc_log = [] 
        self.epoch_times = []
        self.to("mps")

    def row_attention(self, msa_feats):
        B, N, L, d = msa_feats.shape
        x = msa_feats.permute(2, 0, 1, 3).reshape(L, B * N, d)
        out, _ = self.row_attn(x, x, x)
        return out.reshape(L, B, N, d).permute(1, 2, 0, 3)

    def col_attention(self, msa_feats):
        B, N, L, d = msa_feats.shape
        y = msa_feats.reshape(B * N, L, d).permute(1, 0, 2)
        out, _ = self.col_attn(y, y, y)
        return out.permute(1, 0, 2).reshape(B, N, L, d)

    def outer_product_mean(self, msa_feats):
        B, N, L, C = msa_feats.shape
        m_norm = self.layer_norm(msa_feats)
        a = self.transition_a(m_norm)
        b = self.transition_b(m_norm)
        a_mean = a.mean(dim=1)
        b_mean = b.mean(dim=1)
        oij = a_mean.unsqueeze(2) * b_mean.unsqueeze(1)
        # frigör mellanliggande tensorer
        del a_mean, b_mean
        torch.mps.empty_cache() 
        z = self.outer_linear(oij)
        # frigör oij
        del oij
        torch.mps.empty_cache() 
        return z

    def forward(self, seq_ids, msa_ids):
        seq_ids = seq_ids.long()
        msa_ids = msa_ids.long()

        x = self.seq_embed(seq_ids)
        x = self.pe_seq(x)
        x = self.encoder(x)
        x = self.mlp(x)

        B, N, L = msa_ids.shape
        m = self.msa_embed(msa_ids)
        m = self.pe_msa(m.view(B*N, L, self.d_z)).view(B, N, L, self.d_z)
        m = m + self.row_attention(m) + self.col_attention(m)

        msa_pair = self.outer_product_mean(m)


        # NOTE: DEL Z .....
        z = x.unsqueeze(2) * x.unsqueeze(1)
        torch.mps.empty_cache() 
        z = z + msa_pair
        z = z + self.tri_out(z) 
        torch.mps.empty_cache() # Removed this as it can be slow; only use if memory is critical.
        z = z + self.tri_in(z)
        torch.mps.empty_cache() # Removed this as it can be slow; only use if memory is critical.
        
        # CHANGED: Output logits for each bin
        logits = self.pair_mlp(z) # Shape: (B, L, L, nr_bins)
        
        # CHANGED: Enforce symmetric distogram by averaging logits for (i,j) and (j,i)
        # Ensure the labels are also symmetric or only calculate loss on upper triangle.
        logits = 0.5 * (logits + logits.transpose(1, 2)) 
        
        return logits # Returns logits, not probabilities or actual distances

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learningRate)
        # scheduler = StepLR(optimizer, step_size=50, gamma=1)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer


    def OneStep(self, batch):
        seq_ids, msa_ids, labels = batch
        # labels here should be the **integer bin indices** (0 to nr_bins-1)
        # You need to ensure your data loading pipeline converts continuous distances to bin indices.

        logits = self.forward(seq_ids, msa_ids) # Shape: (B, L, L, nr_bins)

        B, L, _, nr_bins = logits.shape
        
        # CHANGED: Create a mask to ignore self-interaction (diagonal)
        # CrossEntropyLoss expects target shape (N) and input shape (N, C)
        # where N is number of samples and C is number of classes.
        
        # Create a boolean mask for valid pairs (excluding diagonal)
        mask = (1 - torch.eye(L, device=logits.device)).bool() # Shape: (L, L)
        
        # Expand the mask to cover the batch dimension and flatten
        mask_flat = mask.unsqueeze(0).expand(B, -1, -1).reshape(-1) # Shape: (B*L*L)

        # Flatten logits and labels, then apply the mask
        logits_flat = logits.view(-1, nr_bins) # Shape: (B*L*L, nr_bins)
        # Ensure labels are long integers for CrossEntropyLoss
        labels_flat = labels.view(-1).long() # Shape: (B*L*L)

        masked_logits = logits_flat[mask_flat]
        masked_labels = labels_flat[mask_flat]

        # Calculate CrossEntropyLoss
        mean_loss = self.loss(masked_logits, masked_labels)

        # CHANGED: Calculate accuracy as the primary metric for classification
        _, predicted_bins = torch.max(masked_logits, 1) # Get the bin with the highest logit
        correct_predictions = (predicted_bins == masked_labels).sum().item()
        # Avoid division by zero if masked_labels is empty
        accuracy = correct_predictions / masked_labels.size(0) if masked_labels.size(0) > 0 else 0 
        
        return mean_loss, accuracy # Return accuracy instead of RMSE

    def training_step(self, batch, batch_idx=None):
        loss, accuracy = self.OneStep(batch)
        self.log("train_accuracy", accuracy, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True) # It's good practice to log training loss too
        return loss

    def validation_step(self, batch, batch_idx=None):
        loss, accuracy = self.OneStep(batch)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True) # <--- ADDED: Log val_loss
        return loss


    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        # CHANGED: Retrieve accuracy metrics
        train_accuracy = self.trainer.callback_metrics.get("train_accuracy", torch.tensor(0.0))
        val_accuracy = self.trainer.callback_metrics.get("val_accuracy", torch.tensor(0.0))
        
        train_loss = self.trainer.callback_metrics.get("train_loss", torch.tensor(0.0)) 
        val_loss = self.trainer.callback_metrics.get("val_loss", torch.tensor(0.0))

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # CHANGED: Log accuracy instead of RMSE
        self.train_acc_log.append(train_accuracy.item()) 
        self.val_acc_log.append(val_accuracy.item()) 
        
        if self.current_epoch % self.print_every == 0:
            eta = sum(self.epoch_times)/len(self.epoch_times) * (self.trainer.max_epochs - self.current_epoch - 1)
            # CHANGED: Print accuracy instead of RMSE
            print(f"Epoch {self.current_epoch}: Train Accuracy {train_accuracy:.4f}, Train Loss {train_loss:.4f}, Val Accuracy {val_accuracy:.4f}, Val Loss {val_loss:.4f}, LR {lr:.2e}")
            print(f"ETA: {eta/60:.1f} min")