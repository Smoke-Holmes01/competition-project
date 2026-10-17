import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
import torch
from tqdm import tqdm

from app.models.encoder import RetrievalEncoder


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrievalEncoder(backbone_name=args.backbone, embedding_dim=args.embedding_dim, pretrained=False)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    features = []
    obj_ids = []

    renders_root = Path(args.renders_root)
    obj_dirs = sorted([p for p in renders_root.iterdir() if p.is_dir()], key=lambda x: int(x.name))
    for obj_dir in tqdm(obj_dirs, desc="Build feature DB"):
        obj_id = int(obj_dir.name)
        view_feats = []
        for img_path in sorted(obj_dir.glob("*.png")):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (args.image_size, args.image_size)).astype(np.float32) / 255.0
            x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(x)[0].cpu().numpy().astype(np.float32)
            view_feats.append(feat)
        if view_feats:
            obj_feat = np.mean(np.stack(view_feats), axis=0)
            obj_feat /= (np.linalg.norm(obj_feat) + 1e-8)
            features.append(obj_feat)
            obj_ids.append(obj_id)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_path, features=np.stack(features), obj_ids=np.array(obj_ids))
    print("特征库已保存到:", args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders_root", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--embedding_dim", type=int, default=256)
    args = parser.parse_args()
    main(args)
