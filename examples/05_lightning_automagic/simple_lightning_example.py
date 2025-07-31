#!/usr/bin/env python3
"""
Simple PyTorch Lightning Example with Tracelet
==============================================

The easiest way to add experiment tracking to your Lightning code.
Just add 3 lines of code to get automatic metric tracking!

This example trains a simple neural network on synthetic data
and automatically logs all metrics to your chosen backend.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# The magic 3 lines to add tracking to ANY Lightning code! ✨
from tracelet import Experiment

exp = Experiment(name="my_lightning_model", backend=["wandb"], automagic=True)
exp.start()


class SimpleModel(pl.LightningModule):
    """A simple feedforward network - your existing Lightning code stays the same!"""

    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Your normal Lightning logging - automatically captured!
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", (logits.argmax(1) == y).float().mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # These metrics are automatically sent to W&B/MLflow/ClearML!
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", (logits.argmax(1) == y).float().mean(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def create_dummy_data():
    """Create some synthetic data for demonstration"""
    # Generate random data
    X = torch.randn(1000, 10)
    y = (X.sum(dim=1) > 0).long()  # Simple binary classification

    # Split into train/val
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return (DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(val_dataset, batch_size=32))


if __name__ == "__main__":
    # Create data
    train_loader, val_loader = create_dummy_data()

    # Create model - your code unchanged!
    model = SimpleModel()

    # Train with Lightning - metrics automatically tracked!
    trainer = pl.Trainer(
        max_epochs=5,
        logger=False,  # Tracelet handles logging for you
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # That's it! Check your W&B/MLflow/ClearML dashboard for results
    exp.stop()

    print("\n✅ Training complete! Check your experiment tracking dashboard.")
    print("   Your metrics have been automatically logged throughout training!")
