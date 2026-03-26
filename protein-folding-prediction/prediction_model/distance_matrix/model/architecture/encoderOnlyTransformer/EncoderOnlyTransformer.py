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

class EncoderOnlyTransformer(L.LightningModule):
    def __init__(
        self,
        maxLengthProtein,
        d_model,
        printEveryEpoch,
        nrOfHeads,
        dropoutRate,
        learningRate,
        nrOfTEncoderLayers,
        d_z
    ):
        super().__init__()
        self.L = maxLengthProtein
        self.d_model = d_model
        self.d_z = d_z
        self.nrOfHeads = nrOfHeads

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
        self.pair_mlp = nn.Sequential(nn.Linear(d_z, 1), nn.Softplus())

        self.loss = nn.MSELoss(reduction='none')
        self.learningRate = learningRate
        self.print_every = printEveryEpoch
        self.train_rmse_log = []
        self.val_rmse_log = []
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

        z = x.unsqueeze(2) * x.unsqueeze(1)
        z = z + msa_pair
        z = z + self.tri_out(z)
        torch.mps.empty_cache()
        z = z + self.tri_in(z)
        torch.mps.empty_cache()
        d_mat = self.pair_mlp(z).squeeze(-1)
        return 0.5 * (d_mat + d_mat.transpose(1, 2))

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learningRate)
        # scheduler = StepLR(optimizer, step_size=50, gamma=1)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer


    def OneStep(self, batch):
        seq_ids, msa_ids, labels = batch
        output = self.forward(seq_ids, msa_ids)
        mask = 1 - torch.eye(output.size(1), device=output.device)
        loss_mat = self.loss(output, labels) * mask
        mean_loss = loss_mat.sum() / (loss_mat != 0).sum()
        rmse = torch.sqrt(mean_loss)
        return mean_loss, rmse

    def training_step(self, batch, batch_idx=None):
        loss, rmse = self.OneStep(batch)
        self.log("train_rmse", rmse, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        loss, rmse = self.OneStep(batch)
        self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)
        return loss


    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        train_rmse = self.trainer.callback_metrics["train_rmse"]
        val_rmse = self.trainer.callback_metrics["val_rmse"]
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.train_rmse_log.append(train_rmse.item())
        self.val_rmse_log.append(val_rmse.item())
        if self.current_epoch % self.print_every == 0:
            eta = sum(self.epoch_times)/len(self.epoch_times) * (self.trainer.max_epochs - self.current_epoch - 1)
            print(f"Epoch {self.current_epoch}: Train RMSE {train_rmse:.4f}, Val RMSE {val_rmse:.4f}, LR {lr:.2e}")
            print(f"ETA: {eta/60:.1f} min")
