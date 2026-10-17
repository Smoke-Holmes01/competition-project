import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.datasets import SegmentationDataset
from app.models.unet import UNet
from app.utils.metrics import dice_score, iou_score


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = 1 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        return bce + dice


def evaluate(model, loader, device):
    model.eval()
    ious, dices = [], []
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            ious.append(iou_score(probs, masks))
            dices.append(dice_score(probs, masks))
    return sum(ious) / len(ious), sum(dices) / len(dices)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SegmentationDataset(args.data_root, args.mask_root, Path(args.splits_dir) / "train.txt", image_size=args.image_size, train=True)
    val_ds = SegmentationDataset(args.data_root, args.mask_root, Path(args.splits_dir) / "val.txt", image_size=args.image_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = BCEDiceLoss()

    best_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        val_iou, val_dice = evaluate(model, val_loader, device)
        print(f"[Val] IoU={val_iou:.4f} Dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print("保存最佳分割模型 ->", args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--mask_root", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
