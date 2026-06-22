"""
download model weights and sample images for axflow examples.

most models are downloaded as .pt from ultralytics releases and exported to
onnx via the ultralytics package. requires `uv add ultralytics`.

usage:
    uv run python scripts/download.py yolov8s
    uv run python scripts/download.py yolov8s-seg
    uv run python scripts/download.py sample-image
    uv run python scripts/download.py all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

REPO_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "assets" / "models"
IMAGES_DIR = REPO_ROOT / "assets" / "images"

# ultralytics .pt → onnx export targets
ULTRA_MODELS = {
    "yolov8s":      "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
    "yolov8s-seg":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt",
}

# direct downloads (not exported)
DIRECT = {
    "sample-image": {
        "url":  "https://ultralytics.com/images/bus.jpg",
        "dest": IMAGES_DIR / "bus.jpg",
        "note": "ultralytics standard test image",
    },
}

def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] already present: {dest.relative_to(REPO_ROOT)}")
        return

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    columns = [
        TextColumn("[bold blue]{task.fields[name]}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*columns) as progress:
        task = progress.add_task("download", total=total, name=dest.name)
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                handle.write(chunk)
                progress.update(task, advance=len(chunk))
    print(f"[done] {dest.relative_to(REPO_ROOT)}")

def export_ultralytics(name: str, url: str) -> None:
    """download .pt and export to onnx via ultralytics package."""
    pt_path   = MODELS_DIR / f"{name}.pt"
    onnx_path = MODELS_DIR / f"{name}.onnx"

    if onnx_path.exists():
        print(f"[skip] already present: {onnx_path.relative_to(REPO_ROOT)}")
        return

    download_file(url, pt_path)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics not installed. run:\n"
            "  uv add ultralytics",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[export] {pt_path.name} → onnx")
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", opset=13, simplify=True)

    exported = Path(exported)
    if exported != onnx_path:
        exported.rename(onnx_path)
    print(f"[done] {onnx_path.relative_to(REPO_ROOT)}")

def fetch(target: str) -> None:
    if target in ULTRA_MODELS:
        export_ultralytics(target, ULTRA_MODELS[target])
    elif target in DIRECT:
        item = DIRECT[target]
        print(f"-> {target}: {item['note']}")
        download_file(item["url"], item["dest"])
    else:
        available = list(ULTRA_MODELS) + list(DIRECT)
        print(f"unknown target '{target}'. available: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)

def fetch_all() -> None:
    for key in list(ULTRA_MODELS) + list(DIRECT):
        fetch(key)

def main() -> None:
    available = list(ULTRA_MODELS) + list(DIRECT) + ["all"]
    parser = argparse.ArgumentParser(description="download axflow assets")
    parser.add_argument("target", choices=available, help="which asset to download")
    args = parser.parse_args()

    if args.target == "all":
        fetch_all()
    else:
        fetch(args.target)

if __name__ == "__main__":
    main()