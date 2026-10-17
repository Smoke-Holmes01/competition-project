import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.models.encoder import RetrievalEncoder
from app.datasets import ClassificationDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ClassificationDataset(
        raw_root=Path("data/raw"),
        image_size=256,
        train=True
    )

    num_classes = len(dataset.groups)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    model = RetrievalEncoder(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/classifier.pth")
    print("Saved: checkpoints/classifier.pth")


if __name__ == "__main__":
    main()
