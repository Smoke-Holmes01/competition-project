import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import cv2
from PIL import Image
from collections import Counter

from app.models.encoder import RetrievalEncoder
from app.models.unet import UNet
from app.retrieval.search import RetrievalSystem


def segment_image(seg_model, image, device):
    seg_model.eval()

    image_resized = cv2.resize(image, (256, 256))
    img_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = seg_model(img_tensor)
        mask = torch.sigmoid(pred)[0, 0].cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    return mask


def crop_by_mask(image, mask):
    coords = np.where(mask > 0)

    if len(coords[0]) == 0:
        return image  # fallback

    y1, y2 = coords[0].min(), coords[0].max()
    x1, x2 = coords[1].min(), coords[1].max()

    cropped = image[y1:y2, x1:x2]

    return cropped


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # 1️⃣ 加载分割模型
    # ======================
    #seg_model = UNet()
    #seg_model.load_state_dict(torch.load("checkpoints/seg_best.pth"))
    #seg_model.to(device)

    # ======================
    # 2️⃣ 加载检索模型
    # ======================
    model = RetrievalEncoder(
        backbone_name="resnet18",
        embedding_dim=256,
        pretrained=False
    )

    state_dict = torch.load("checkpoints/retrieval_best.pth")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # ======================
    # 3️⃣ 检索系统
    # ======================
    system = RetrievalSystem(
        model,
        "faiss_index/index.faiss",
        "faiss_index/metadata.pkl",
        device
    )

    # ======================
    # 4️⃣ 输入图片
    # ======================
    img_path = "data/test/1.png"

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # ======================
    # ⭐ 分割
    # ======================
    #mask = segment_image(seg_model, image, device)

    # ======================
    # ⭐ 裁剪
    # ======================
    cropped = image  # 直接用原图

    # 可选：保存看看效果
    #cv2.imwrite("debug_mask.png", mask)
    cv2.imwrite("debug_crop.png", cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

    # ======================
    # ⭐ 转 tensor
    # ======================
    cropped = cv2.resize(cropped, (256, 256))
    tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    # ======================
    # ⭐ 检索
    # ======================
    results = system.search(tensor, topk=10)

    print("\nQuery:", img_path)
    print("Top10 results:")
    for r in results:
        print(r)

    # ======================
    # ⭐ 投票
    # ======================
    groups = [p.split("\\")[-2] for p in results]
    final = Counter(groups).most_common(1)[0][0]

    print("\nFinal match:", final)


if __name__ == "__main__":
    main()
