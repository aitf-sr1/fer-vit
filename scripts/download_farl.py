#!/usr/bin/env python3
import sys
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "ep16": {
        "url": "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep16.pth",
        "filename": "FaRL-Base-Patch16-LAIONFace20M-ep16.pth",
        "description": "FaRL ViT-B/16, trained 16 epochs (used in paper)",
    },
    "ep64": {
        "url": "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
        "filename": "FaRL-Base-Patch16-LAIONFace20M-ep64.pth",
        "description": "FaRL ViT-B/16, trained 64 epochs (stronger)",
    },
}

DEFAULT_OUT_DIR = Path("outputs/farl")


def download(variant: str = "ep16", out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    ckpt = CHECKPOINTS[variant]
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / ckpt["filename"]

    if dest.exists():
        print(f"Checkpoint already exists: {dest}")
        return dest

    print(f"Downloading {ckpt['description']}...")
    print(f"  URL: {ckpt['url']}")
    print(f"  Destination: {dest}")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            mb = count * block_size / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {pct:.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(ckpt["url"], dest, reporthook=_progress)
    print(f"\nSaved to: {dest}")
    return dest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FaRL pretrained checkpoint")
    parser.add_argument(
        "--variant",
        choices=["ep16", "ep64"],
        default="ep16",
        help="Which checkpoint to download (default: ep16)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    path = download(args.variant, Path(args.out_dir))
    print(f'\nAdd to farl_binary.yaml:\n  farl_checkpoint: "{path}"')
