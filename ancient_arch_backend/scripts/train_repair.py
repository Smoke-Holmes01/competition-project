from pathlib import Path
import sys
import random
import csv

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.datasets_repair import RepairDataset
from app.models.repair_unet import RepairUNet


def get_group_names(root_dir: str):
    root = Path(root_dir)
    groups = sorted([
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.startswith("group_")
    ])
    return groups


def split_groups(group_names, val_ratio=0.1, seed=42):
    random.seed(seed)
    groups = group_names.copy()
    random.shuffle(groups)

    val_count = max(1, int(len(groups) * val_ratio))
    val_groups = groups[:val_count]
    train_groups = groups[val_count:]

    return train_groups, val_groups


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()

    return total_loss / len(loader)


def tensor_to_image(tensor):
    """
    tensor: [C, H, W], range [0,1]
    -> uint8 RGB image
    """
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def save_visualization(model, loader, device, save_path: Path, max_samples=4):
    """
    保存验证样本可视化:
    damaged | pred | gt
    """
    model.eval()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            batch_size = x.size(0)
            for i in range(min(batch_size, max_samples - len(rows))):
                damaged_img = tensor_to_image(x[i])
                pred_img = tensor_to_image(pred[i])
                gt_img = tensor_to_image(y[i])

                row = np.concatenate([damaged_img, pred_img, gt_img], axis=1)
                rows.append(row)

                if len(rows) >= max_samples:
                    break

            if len(rows) >= max_samples:
                break

    if len(rows) > 0:
        canvas = np.concatenate(rows, axis=0)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), canvas_bgr)


def append_log(csv_path: Path, epoch: int, train_loss: float, val_loss: float):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])


def train():
    # ===== 基本配置 =====
    root_dir = "data/raw"
    image_size = 256
    batch_size = 8
    epochs = 5
    lr = 1e-3
    val_ratio = 0.1
    seed = 42
    resume = True  # 是否断点续训

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_model_path = checkpoint_dir / "repair_unet.pth"
    log_csv_path = checkpoint_dir / "repair_train_log.csv"
    vis_dir = checkpoint_dir / "repair_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # ===== 数据划分 =====
    all_groups = get_group_names(root_dir)
    if len(all_groups) == 0:
        raise ValueError(f"在 {root_dir} 下没有找到 group_xxx 目录")

    train_groups, val_groups = split_groups(all_groups, val_ratio=val_ratio, seed=seed)

    print("Train groups:", train_groups)
    print("Val groups:", val_groups)

    train_dataset = RepairDataset(
        root_dir=root_dir,
        groups=train_groups,
        image_size=image_size
    )
    val_dataset = RepairDataset(
        root_dir=root_dir,
        groups=val_groups,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # ===== 模型 =====
    model = RepairUNet(in_channels=3, out_channels=3).to(device)

    if resume and save_model_path.exists():
        model.load_state_dict(torch.load(str(save_model_path), map_location=device))
        print(f"Loaded existing checkpoint for resume: {save_model_path}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    # 如果是续训，可以尝试从日志恢复 best_val_loss
    if log_csv_path.exists():
        try:
            with open(log_csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
                if len(rows) > 1:
                    val_losses = [float(r[2]) for r in rows[1:] if len(r) >= 3]
                    if len(val_losses) > 0:
                        best_val_loss = min(val_losses)
                        print(f"Recovered best_val_loss={best_val_loss:.6f} from history")
        except Exception as e:
            print(f"Failed to read history log, best_val_loss will be reset: {e}")

    # ===== 开始训练 =====
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # ===== 验证 =====
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # ===== 记录日志 =====
        append_log(log_csv_path, epoch, train_loss, val_loss)

        # ===== 保存可视化 =====
        vis_path = vis_dir / f"epoch_{epoch:03d}.png"
        save_visualization(model, val_loader, device, vis_path, max_samples=4)
        print(f"Saved visualization: {vis_path}")

        # ===== 保存最佳模型 =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_model_path)
            print(f"Saved best model to: {save_model_path}")

    print("Training complete")
    print(f"最佳模型路径: {save_model_path}")
    print(f"训练日志路径: {log_csv_path}")
    print(f"可视化结果目录: {vis_dir}")


if __name__ == "__main__":
    train()
