import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.models.encoder import RetrievalEncoder
from app.retrieval.build_index import build_faiss_index
from app.datasets import list_group_images, build_default_transform


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # ✅ 加载模型
    # ======================
    model = RetrievalEncoder(
        backbone_name="resnet18",
        embedding_dim=256,
        pretrained=False
    )

    state_dict = torch.load("checkpoints/retrieval_best.pth")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # ======================
    # ⭐关键：用 render 建库
    # ======================
    render_root = Path("data/renders")

    image_paths = []
    for gid_dir in render_root.iterdir():
        if gid_dir.is_dir():
            for img_path in list_group_images(gid_dir):
                image_paths.append(img_path)

    print("Render image count:", len(image_paths))

    # ======================
    # 提取 embedding
    # ======================
    transform = build_default_transform(256, train=False)

    embeddings = []
    valid_paths = []

    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image = transform(image=np.array(image))["image"].unsqueeze(0).to(device)

            emb = model(image)
            embeddings.append(emb.cpu())
            valid_paths.append(str(path))

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.numpy()

    # ======================
    # 建立 FAISS index
    # ======================
    build_faiss_index(embeddings, valid_paths, "faiss_index")

    print("Render-based FAISS index built")


if __name__ == "__main__":
    main()
