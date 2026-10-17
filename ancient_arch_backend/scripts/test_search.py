import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from PIL import Image
from torchvision import transforms

from app.models.encoder import RetrievalEncoder
from app.retrieval.search import RetrievalSystem


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 加载模型
    model = RetrievalEncoder(
        backbone_name="resnet18",
        embedding_dim=256,
        pretrained=False
    )

    state_dict = torch.load("checkpoints/retrieval_best.pth")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # ✅ 初始化检索系统
    system = RetrievalSystem(
        model,
        "faiss_index/index.faiss",
        "faiss_index/metadata.pkl",
        device
    )

    # ✅ 读取测试图片（改成你自己的路径！！）
    img_path = "data/test/1.png"   # ⭐这里改成你的一张图

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # ✅ 搜索
    results = system.search(image, topk=5)

    print("\nQuery image:", img_path)
    print("Most similar results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
