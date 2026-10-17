import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.preprocess import read_image
from app.services.segment_service import segment_service
from app.services.repair_service import repair_service
from app.services.retrieval_service import retrieval_service
from app.services.ar_service import ar_service


def main(args):
    image = read_image(Path(args.image))
    mask_path, crop_path, _, crop, method, _ = segment_service.segment_and_save(image, Path(args.image).stem)
    print("分割crop:", crop_path)
    print("分割方法:", method)

    repaired_path, repaired_img = repair_service.repair(crop)
    query = repaired_img if repaired_img is not None else crop
    if repaired_path:
        print("修复结果:", repaired_path)

    topk = retrieval_service.search(query, topk=args.topk)
    print("TopK:", topk)

    best = topk[0]
    print("匹配obj:", best["obj_id"])
    print("模型路径:", ar_service.get_model_path(best["obj_id"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    main(args)
