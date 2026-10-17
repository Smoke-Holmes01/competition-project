import faiss
import pickle
import torch


class RetrievalSystem:
    def __init__(self, model, index_path, metadata_path, device):
        self.model = model.to(device)
        self.model.eval()

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.device = device

    def search(self, image_tensor, topk=5):
        with torch.no_grad():
            emb = self.model(image_tensor.to(self.device))
            emb = emb.cpu().numpy()

        D, I = self.index.search(emb, topk)

        results = [self.metadata[i] for i in I[0]]

        return results