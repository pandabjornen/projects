import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

def trainBINS(model, trainDataLoader, valDataLoader, modelPath, nrOfEpochs, accumBatches):
    model.to("mps")

    # Change 'val_rmse' to 'val_loss' to match the logged metric
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",        # Now monitoring validation loss
        mode="min",                # Save model with the lowest validation loss
        save_top_k=1,              # Save only the best model
        dirpath= modelPath,
        filename="{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}" # Updated filename to reflect new metrics
    )
    
    trainer = L.Trainer(accelerator="mps", accumulate_grad_batches=accumBatches,
        max_epochs=nrOfEpochs,
        callbacks=[checkpoint_cb],
        enable_progress_bar=False,
        logger=False
    )
    trainer.fit(model, train_dataloaders=trainDataLoader, val_dataloaders=valDataLoader) 
    return model