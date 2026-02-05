import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time

DATA_PATH = "../processed/train_transformed.csv"
BATCH_SIZE = 1024
EMBED_DIM = 64
EPOCHS = 10
LR = 1e-3
DEVICE = "cpu"

PRINT_EVERY = 200   # batches

class RatingsDataset(Dataset):
    def __init__(self, csv_path):
        print("Reading CSV...")
        start = time.time()

        df = pd.read_csv(csv_path)

        print(f"CSV loaded in {time.time() - start:.2f}s")
        print("Converting columns to tensors...")

        self.users = torch.tensor(df["userId"].values, dtype=torch.long)
        self.items = torch.tensor(df["movieId"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

        self.num_users = self.users.max().item() + 1
        self.num_items = self.items.max().item() + 1

        print(f"Dataset ready | Rows: {len(self.ratings)}")
        print(f"Users: {self.num_users} | Items: {self.num_items}")

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        self.user_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        self.item_tower = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, users, items):
        u = self.user_embedding(users)
        i = self.item_embedding(items)

        u = self.user_tower(u)
        i = self.item_tower(i)

        return (u * i).sum(dim=1)

def main():
    print("========== SETUP ==========")
    dataset = RatingsDataset(DATA_PATH)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    print(f"DataLoader batches per epoch: {len(train_loader)}")

    model = TwoTowerModel(
        dataset.num_users,
        dataset.num_items,
        EMBED_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\n========== TRAINING ==========\n")

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} START ---")
        epoch_start = time.time()

        total_loss = 0.0

        for batch_idx, (users, items, ratings) in enumerate(train_loader):
            if batch_idx == 0:
                print("First batch received")

            users = users.to(DEVICE)
            items = items.to(DEVICE)
            ratings = ratings.to(DEVICE)

            preds = model(users, items)
            loss = loss_fn(preds, ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % PRINT_EVERY == 0:
                elapsed = time.time() - epoch_start
                print(
                    f"Epoch {epoch+1} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f} | "
                    f"Elapsed {elapsed:.1f}s"
                )

        avg_loss = total_loss / len(train_loader)
        print(
            f"--- Epoch {epoch + 1} DONE | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Time: {time.time() - epoch_start:.1f}s ---"
        )

    print("\n========== DONE ==========")
    torch.save(model.state_dict(), "../artifacts/twotower_model.pt")
    print("Model saved as twotower_model.pt")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
