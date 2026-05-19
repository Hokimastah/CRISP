from __future__ import annotations

from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def list_images(folder: str) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def infer_label_from_parent(path: Path) -> str:
    return path.parent.name
