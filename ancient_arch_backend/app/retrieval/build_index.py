import faiss
import numpy as np
import pickle
from pathlib import Path


def build_faiss_index(embeddings, paths, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)  # ⭐配合 normalize
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(save_dir / "index.faiss"))

    with open(save_dir / "metadata.pkl", "wb") as f:
        pickle.dump(paths, f)

    print("FAISS index built:", embeddings.shape)
