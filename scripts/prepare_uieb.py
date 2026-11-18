"""Record an existing UIEB split or rebuild it from a pair of filename lists."""

import argparse
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def index_images(folder):
    images = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in images:
                raise ValueError(f"Duplicate image stem in {folder}: {path.stem}")
            images[path.stem] = path
    if not images:
        raise ValueError(f"No images in {folder}")
    return images


def export_split(root, output):
    groups = {}
    for split in ("train", "val"):
        inputs = index_images(root / split / "lq")
        targets = index_images(root / split / "gt")
        if inputs.keys() != targets.keys():
            raise ValueError(f"Unpaired images in {split}")
        groups[split] = set(inputs)
        (output / f"{split}.txt").write_text(
            "".join(inputs[key].name + "\n" for key in sorted(inputs)), encoding="utf-8"
        )
    if groups["train"] & groups["val"]:
        raise ValueError("Training and validation have overlapping image stems.")


def prepare(raw, reference, lists, output):
    inputs, targets = index_images(raw), index_images(reference)
    seen = set()
    for split in ("train", "val"):
        lines = (lists / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines()
        names = [name.strip() for name in lines if name.strip()]
        if not names:
            raise ValueError(f"Empty {split} list")
        for kind in ("lq", "gt"):
            (output / split / kind).mkdir(parents=True)
        for name in names:
            key = Path(name).stem
            if key in seen:
                raise ValueError(f"Duplicate or overlapping image: {name}")
            if key not in inputs or key not in targets or inputs[key].name != name:
                raise ValueError(f"Missing pair for {name}")
            seen.add(key)
            copy2(inputs[key], output / split / "lq")
            copy2(targets[key], output / split / "gt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    export = modes.add_parser("export")
    export.add_argument("--root", required=True, type=Path)
    export.add_argument("--output", default=Path("splits"), type=Path)
    build = modes.add_parser("prepare")
    build.add_argument("--raw", required=True, type=Path)
    build.add_argument("--reference", required=True, type=Path)
    build.add_argument("--splits", default=Path("splits"), type=Path)
    build.add_argument("--output", default=Path("data/UIEB"), type=Path)
    args = parser.parse_args()
    dest = args.output
    if dest.exists():
        parser.error(f"Choose a new destination: {dest} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as temp:
        staged = Path(temp) / "output"
        staged.mkdir()
        if args.mode == "export":
            export_split(args.root, staged)
        else:
            prepare(args.raw, args.reference, args.splits, staged)
        staged.rename(dest)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
