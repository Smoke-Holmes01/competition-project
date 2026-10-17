from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert OBJ assets to GLB for web delivery.")
    parser.add_argument("--source-dir", type=Path, default=Path("models"))
    parser.add_argument("--target-dir", type=Path, default=Path("models_glb"))
    parser.add_argument("--expected-count", type=int, default=46)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_sort_key(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Unexpected OBJ filename: {path.name}") from exc


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir

    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    obj_files = sorted(source_dir.glob("*.obj"), key=model_sort_key)
    if len(obj_files) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} OBJ files, found {len(obj_files)} in {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for obj_path in obj_files:
        glb_path = target_dir / f"{obj_path.stem}.glb"
        if glb_path.exists() and not args.overwrite:
            continue

        try:
            scene = trimesh.load(obj_path, force="scene", process=False)
            glb_bytes = scene.export(file_type="glb")
            glb_path.write_bytes(glb_bytes)
        except Exception as exc:
            failures.append(f"{obj_path.name}: {exc}")

    generated = sorted(target_dir.glob("*.glb"), key=model_sort_key)
    expected_names = {f"{index}.glb" for index in range(1, args.expected_count + 1)}
    actual_names = {path.name for path in generated}
    missing = sorted(expected_names - actual_names, key=lambda name: int(Path(name).stem))

    if failures or missing:
        details = []
        if failures:
            details.append("Failures: " + "; ".join(failures))
        if missing:
            details.append("Missing GLB files: " + ", ".join(missing))
        raise SystemExit("GLB conversion did not complete. " + " ".join(details))

    print(f"Converted/verified {args.expected_count} GLB files in {target_dir}")


if __name__ == "__main__":
    main()
