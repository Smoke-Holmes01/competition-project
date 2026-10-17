import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.datasets import TripletGroupDataset
from app.models.encoder import RetrievalEncoder


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TripletGroupDataset(args.data_root, Path(args.splits_dir) / "train.txt", args.renders_root, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = RetrievalEncoder(backbone_name=args.backbone, embedding_dim=args.embedding_dim, pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.TripletMarginLoss(margin=0.3, p=2)

    best_loss = 1e9
    for epoch in range(args.epochs):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            loss = criterion(emb_a, emb_p, emb_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        mean_loss = sum(losses) / max(len(losses), 1)
        print(f"Epoch {epoch + 1}: mean loss = {mean_loss:.4f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print("保存最佳检索模型 ->", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--renders_root", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--embedding_dim", type=int, default=256)
    args = parser.parse_args()
    main(args)
