import torch as t
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import ChallengeDataset
from trainer import Trainer
import model

# ----------------------------
# Hyperparameters
# ----------------------------
BATCH_SIZE = 32          # try 64 if VRAM allows
LEARNING_RATE = 3e-4     # gentler LR tends to stabilize F1
EPOCHS = 50              # ceiling; early stop on F1 will stop earlier
VAL_SPLIT = 0.2
SEED = 42


def main():
    t.manual_seed(SEED)

    # ----------------------------
    # 1) Load and split the data
    # ----------------------------
    data_df = pd.read_csv("data.csv", sep=";")
    train_df, val_df = train_test_split(
        data_df, test_size=VAL_SPLIT, random_state=SEED, shuffle=True
    )
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # ----------------------------
    # 2) Create datasets
    # ----------------------------
    train_dataset = ChallengeDataset(train_df, mode="train")
    val_dataset   = ChallengeDataset(val_df, mode="val")

    # ----------------------------
    # 3) DataLoaders (GPU-friendly)
    # ----------------------------
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print("Using", "GPU" if t.cuda.is_available() else "CPU")
    if t.cuda.is_available():
        t.set_float32_matmul_precision("high")

    # ----------------------------
    # 4) Model, loss, optimizer
    # ----------------------------
    net = model.ResNet()
    criterion = t.nn.BCELoss()  # model outputs are sigmoid probabilities
    optimizer = t.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Optional scheduler (reduces LR when val loss plateaus)
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # ----------------------------
    # 5) Trainer
    # ----------------------------
    trainer = Trainer(net, criterion, cuda=t.cuda.is_available())
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.early_stopping_patience = 6

    # ----------------------------
    # 6) Train (per-epoch F1 printed by Trainer.fit)
    # ----------------------------
    train_losses, val_losses = trainer.fit(
        train_loader, val_loader, epochs=EPOCHS, verbose=True
    )

    # ----------------------------
    # 7) Plot and save loss curves
    # ----------------------------
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses, label="Train Loss")
    plt.plot(np.arange(len(val_losses)), val_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("losses.png")
    plt.close()
    print("Training complete. Loss curve saved as 'losses.png'.")


if __name__ == "__main__":
    main()
