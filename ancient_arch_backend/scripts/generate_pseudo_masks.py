import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2

from app.datasets import IMG_EXTS
from app.services.preprocess import read_image, save_gray
from app.services.pseudo_segmentation import generate_pseudo_mask


def iter_images(raw_root: Path):
    for group_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        for image_path in sorted([p for p in group_dir.iterdir() if p.suffix.lower() in IMG_EXTS]):
            yield group_dir.name, image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--mask-root", type=Path, default=Path("data/masks"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    written = 0
    skipped = 0
    for group, image_path in iter_images(args.raw_root):
        if args.limit and written >= args.limit:
            break
        output_path = args.mask_root / group / image_path.name
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        image = read_image(image_path)
        mask = generate_pseudo_mask(image)
        if cv2.countNonZero(mask) == 0:
            raise RuntimeError(f"Generated an empty mask for {image_path}")
        save_gray(mask, output_path)
        written += 1

    print(f"Pseudo masks written: {written}")
    print(f"Existing masks skipped: {skipped}")


if __name__ == "__main__":
    main()
