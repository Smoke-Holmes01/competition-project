import torch
import os

from app.retrieval.clip_system import CLIPRetrieval


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    system = CLIPRetrieval(device)

    # ✅ 收集所有 render 图
    render_root = "data/renders"

    image_paths = []
    for gid in os.listdir(render_root):
        group_dir = os.path.join(render_root, gid)
        for img in os.listdir(group_dir):
            image_paths.append(os.path.join(group_dir, img))

    # ✅ 建库
    system.build_gallery(image_paths)

    # ✅ 查询（换成你的测试图）
    query = "data/test/1.png"

    results = system.search(query, topk=5)

    print("\nQuery:", query)
    print("Results:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
