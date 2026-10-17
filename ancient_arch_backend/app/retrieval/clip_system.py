import torch
from PIL import Image


class CLIPRetrieval:
    def __init__(self, device):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("CLIPRetrieval requires optional dependency open_clip_torch") from exc

        self.device = device

        # ⭐加载CLIP模型（很强）
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        self.model = self.model.to(device)
        self.model.eval()

        self.gallery_embeddings = []
        self.gallery_paths = []

    def build_gallery(self, image_paths):
        embeddings = []

        with torch.no_grad():
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                image = self.preprocess(image).unsqueeze(0).to(self.device)

                emb = self.model.encode_image(image)
                emb = emb / emb.norm(dim=-1, keepdim=True)

                embeddings.append(emb.cpu())
                self.gallery_paths.append(path)

        self.gallery_embeddings = torch.cat(embeddings, dim=0)

        print("CLIP gallery built:", self.gallery_embeddings.shape)

    def search(self, query_path, topk=5):
        image = Image.open(query_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            query_emb = self.model.encode_image(image)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

        sims = query_emb @ self.gallery_embeddings.T
        topk_idx = sims[0].topk(topk).indices

        results = [self.gallery_paths[i] for i in topk_idx]

        return results
