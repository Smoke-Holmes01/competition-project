import argparse
import random
from pathlib import Path

from app.utils.io import save_json, write_lines


def main(args):
    random.seed(args.seed)
    data_root = Path(args.data_root)
    groups = sorted([p.name for p in data_root.iterdir() if p.is_dir() and p.name.startswith("group_")])
    random.shuffle(groups)

    n = len(groups)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(train_groups, output_dir / "train.txt")
    write_lines(val_groups, output_dir / "val.txt")
    write_lines(test_groups, output_dir / "test.txt")

    mapping = {}
    for g in sorted(groups):
        gid = int(g.split("_")[-1])
        mapping[g] = {
            "obj_id": gid,
            "obj_name": f"{gid}.obj"
        }
    save_json(mapping, Path(args.mapping_path))
    print("总组数:", n)
    print("train:", len(train_groups), train_groups)
    print("val:", len(val_groups), val_groups)
    print("test:", len(test_groups), test_groups)
    print("映射已保存到:", args.mapping_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mapping_path", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.78)
    parser.add_argument("--val_ratio", type=float, default=0.11)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
