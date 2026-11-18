"""Prepare UIEB training options in a checkout of BinCVER/BEM."""

import argparse
from pathlib import Path
import subprocess

import yaml

REVISION = "ffd10299a2b9677f30f26f8ad9af33b3b4e099bd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", args.checkout, "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        parser.error(f"Check out {REVISION} before preparing the options.")
    for prefix in ("CG", "IE"):
        options = args.checkout / "Options" / f"{prefix}_UNet_UIEB.yml"
        config = yaml.safe_load(options.read_text())
        for split in ("train", "val"):
            for kind in ("lq", "gt"):
                directory = (args.data_root / split / kind).resolve()
                if not directory.is_dir():
                    parser.error(f"Missing dataset directory: {directory}")
                config["datasets"][split]["dataroot_" + kind] = str(directory)
        if prefix == "IE":
            steps = sum(config["train"]["scheduler"]["periods"])
            config["train"]["total_iter"] = steps
            config["datasets"]["train"]["iters"] = [steps]
        dest = options.with_name(options.stem + "_local.yml")
        dest.write_text(yaml.safe_dump(config, sort_keys=False))
        print(dest)


if __name__ == "__main__":
    main()
