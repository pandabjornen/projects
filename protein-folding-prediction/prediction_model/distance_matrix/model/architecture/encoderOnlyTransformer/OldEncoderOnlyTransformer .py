import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD

from distance_matrix.model.architecture.encoderOnlyTransformer.AttentionSelf import Attention
from distance_matrix.model.architecture.encoderOnlyTransformer.PositionalEncodingg import PositionEncoding
from torch.optim.lr_scheduler import ReduceLROnPlateau
from distance_matrix.model.architecture.encoderOnlyTransformer.TransformerEncoderBlock import TransformerEncoderBlock

import lightning as L 
from distance_matrix.model.architecture.encoderOnlyTransformer.TriangleUpdate import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
import time

class EncoderOnlyTransformer(L.LightningModule): 

    def __init__(self, maxLengthProtein, d_model, printEveryEpoch, nrOfHeads, dropoutRate, learningRate, nrOfTEncoderLayers, d_z):
        super().__init__()

        self.L = maxLengthProtein

        self.tri_out = TriangleMultiplicationOutgoing(d_z)
        self.tri_in  = TriangleMultiplicationIncoming(d_z)

        self.train_rmse_log = []
        self.val_rmse_log = []
        self.epoch_times = []

        self.learningRate = learningRate
        self.print_every = printEveryEpoch

        self.we = nn.Embedding(num_embeddings = maxLengthProtein, embedding_dim=d_model) 
        self.pe = PositionEncoding(d_model, maxLengthProtein)

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(d_model, nrOfHeads, dropoutRate)
            for _ in range(nrOfTEncoderLayers) 
        ])

        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_z),
            # nn.ReLU()     # Force output to be >= 0 (distances)  #NOTE: SEEMS to WORSEN ACCURACY THOUGH, EARLY IN TRAINING ATLEAST
                            # NOTE: RELU bad because kills all negative activations, cant learn...

            nn.Softplus()
        )

        self.pair_mlp = nn.Sequential(
            nn.Linear(d_z, 1),
            # nn.ReLU()     # Force output to be >= 0 (distances)  #NOTE: SEEMS to WORSEN ACCURACY THOUGH, EARLY IN TRAINING ATLEAST
                            # NOTE: RELU bad because kills all negative activations, cant learn...

            nn.Softplus()
        )
        self.loss = nn.MSELoss(reduction='none') #elementwise

        self.to("mps")

    def forward(self, token_ids):
        x = self.pe(self.we(token_ids))
        x = self.encoder(x)
        x = self.mlp(x)
        z = x.unsqueeze(2) * x.unsqueeze(1)  # [B, L, L, d_model]
        
        out1 = self.tri_out(z)
        z = z + out1; del out1
        torch.mps.empty_cache()
        out2 = self.tri_in(z)
        z = z + out2; del out2
        torch.mps.empty_cache()


        
        d = self.pair_mlp(z).squeeze(-1)     # [B, L, L]
        
        return 0.5 * (d + d.transpose(1, 2))
  
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=self.learningRate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',      # Reduce LR when the monitored quantity has stopped decreasing
            factor=0.75,      # Factor by which the learning rate will be reduced. new_lr = lr * factor
            patience=20,     # Number of epochs with no improvement after which learning rate will be reduced
            verbose=True,    # If True, prints a message to stdout for each update
            min_lr=1e-6      # A lower bound on the learning rate
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_rmse", # Metric to monitor
                "interval": "epoch",   # Check scheduler conditions every epoch
                "frequency": 1,        # Check every 1 epoch
            },
        }
    
    def OneStep(self, batch):
        input_tokens, labels = batch 
        output = self.forward(input_tokens)
        
        maskDiagonal = torch.eye(output.shape[1], device=output.device)  
        loss = self.loss(output, labels)
        loss = loss * (1 - maskDiagonal)  
        nrElementsNotOnDiagonalAndNotPadding = (loss != 0).sum()

        
        meanLoss = loss.sum() / nrElementsNotOnDiagonalAndNotPadding
        rmse = torch.sqrt(meanLoss)
        return meanLoss, rmse
    
    def training_step(self, batch):
        meanLoss, rmse = self.OneStep(batch)
        self.log("train_rmse", rmse, prog_bar=False, on_epoch=True)
        return meanLoss

    def validation_step(self, batch):
        meanLoss, rmse = self.OneStep(batch)
        self.log("val_rmse", rmse, prog_bar=False, on_step=False, on_epoch=True)
        return meanLoss
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        train_rmse = self.trainer.callback_metrics.get("train_rmse")
        val_rmse = self.trainer.callback_metrics.get("val_rmse")
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        self.train_rmse_log.append(train_rmse.item())
        self.val_rmse_log.append(val_rmse.item())

        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.trainer.max_epochs - self.current_epoch - 1
        eta = avg_time * remaining_epochs

        if self.current_epoch % self.print_every == 0:
            print(f"Epoch {self.current_epoch}: Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, LR: {current_lr:.2e}")
            print(f"Estimated Time Left: {eta/60:.1f} min")



