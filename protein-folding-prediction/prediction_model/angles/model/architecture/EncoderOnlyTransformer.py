import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from angles.model.architecture.PositionalEncodingg import PositionEncoding
from angles.model.architecture.TransformerEncoderBlock import TransformerEncoderBlock
import lightning as L 
import time

class EncoderOnlyTransformerNNAngles(L.LightningModule): 

    def __init__(self, maxLengthProtein, d_model, printEveryEpoch, nrOfHeads, dropoutRate, learningRate, nrOfTEncoderLayers):
        super().__init__()

        self.L = maxLengthProtein

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

        self.project_to_angle = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(), #want to map output to [0,1] and then multiply by 180 to get degrees
        )
        self.loss = nn.MSELoss(reduction='none') #elementwise
        self.to("mps")

    def forward(self, token_ids):
        x = self.pe(self.we(token_ids))
        x = self.encoder(x)
        x = self.project_to_angle(x)
        x = x *180 # -> [0, 180] inre vinkel
        return x
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr = self.learningRate)
    
    def OneStep(self, batch):
        input_tokens, labels = batch 
        output = self.forward(input_tokens)
        
        loss = self.loss(output.squeeze(-1), labels)
        mask = (labels != 0).float()

        loss = loss * mask
        nrElements = mask.sum()

        meanLoss = loss.sum() / nrElements
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

        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        self.train_rmse_log.append(train_rmse.item())
        self.val_rmse_log.append(val_rmse.item())

        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.trainer.max_epochs - self.current_epoch - 1
        eta = avg_time * remaining_epochs

        if self.current_epoch % self.print_every == 0:
            print(f"Epoch {self.current_epoch}: RMSE train: {train_rmse:.4f}° , val: {val_rmse:.4f}° ")
            print(f"Estimated Time Left: {eta/60:.1f} min")



