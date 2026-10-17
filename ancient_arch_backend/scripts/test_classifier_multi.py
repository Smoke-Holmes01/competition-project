import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
import cv2

from app.models.encoder import RetrievalEncoder
from app.datasets import ClassificationDataset


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载类别映射
    dataset = ClassificationDataset(
        raw_root=Path("data/raw"),
        image_size=256,
        train=False
    )
    num_classes = len(dataset.groups)

    # 2. 加载模型
    model = RetrievalEncoder(num_classes=num_classes)
    model.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device))
    model.to(device)
    model.eval()

    # 3. 用户多张图片目录
    test_dir = Path("data/test_multi")
    image_paths = list_images(test_dir)

    if len(image_paths) == 0:
        print("data/test_multi has no images")
        return

    print(f"Detected {len(image_paths)} images")

    # 4. 累加概率
    prob_sum = None

    with torch.no_grad():
        for img_path in image_paths:
            image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            image = dataset.transform(image=image)["image"]
            image = image.unsqueeze(0).to(device)

            logits = model(image)
            probs = torch.softmax(logits, dim=1)

            pred = probs.argmax(dim=1).item()
            pred_group = dataset.label_to_group[pred]
            confidence = probs[0, pred].item()

            print(f"{img_path.name} -> {pred_group}  (confidence: {confidence:.4f})")

            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum += probs

    # 5. 平均概率
    avg_probs = prob_sum / len(image_paths)
    final_pred = avg_probs.argmax(dim=1).item()
    final_group = dataset.label_to_group[final_pred]
    final_conf = avg_probs[0, final_pred].item()

    print("\n" + "=" * 40)
    print(f"Final result: {final_group}")
    print(f"Combined confidence: {final_conf:.4f}")

    # 可选：输出Top5
    topk = torch.topk(avg_probs, k=5, dim=1)
    print("\nTop5 candidates:")
    for score, idx in zip(topk.values[0], topk.indices[0]):
        print(f"{dataset.label_to_group[idx.item()]}  ->  {score.item():.4f}")


if __name__ == "__main__":
    main()
