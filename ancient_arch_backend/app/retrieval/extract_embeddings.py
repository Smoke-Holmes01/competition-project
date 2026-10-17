import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def extract_embeddings(model, dataset, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()

    all_embeddings = []
    all_paths = []

    with torch.no_grad():
        for images, gids, paths in loader:
            images = images.to(device)

            emb = model(images)

            all_embeddings.append(emb.cpu().numpy())
            all_paths.extend(paths)

    embeddings = np.vstack(all_embeddings)

    return embeddings, all_paths