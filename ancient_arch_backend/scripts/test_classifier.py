import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import torch
import cv2

from app.models.encoder import RetrievalEncoder
from app.datasets import ClassificationDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ClassificationDataset(
        raw_root=Path("data/raw"),
        image_size=256,
        train=False
    )

    num_classes = len(dataset.groups)

    model = RetrievalEncoder(num_classes=num_classes)
    model.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device))
    model.to(device)
    model.eval()

    img_path = "data/test/1.png"

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = dataset.transform(image=image)["image"]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        pred = logits.argmax(dim=1).item()

    pred_group = dataset.label_to_group[pred]

    print("Predicted class index:", pred)
    print("Predicted group:", pred_group)


if __name__ == "__main__":
    main()
