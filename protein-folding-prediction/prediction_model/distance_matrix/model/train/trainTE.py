import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint

def trainTE(model, trainDataLoader, valDataLoader, modelPath, nrOfEpochs, accumBatches):
    model.to("mps")

    # i din Trainer:
    checkpoint_cb = ModelCheckpoint(
        monitor="val_rmse",        # metrics‑namnet du loggar
        mode="min",                # vill ha lägsta val_rmse
        save_top_k=1,              # spara bara bästa
        dirpath= modelPath,
        filename="{epoch:02d}-{val_rmse:.4f}"
    )
    # precision="16-mixed",
    trainer = L.Trainer(accelerator="mps", accumulate_grad_batches=accumBatches, max_epochs=nrOfEpochs, callbacks=[checkpoint_cb],enable_progress_bar=False, logger=False)
    trainer.fit(model, train_dataloaders=trainDataLoader, val_dataloaders= valDataLoader) 
    return model
