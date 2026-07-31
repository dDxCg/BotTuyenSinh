"""Tải multilingual-e5-large về workspace để query embedding chạy local."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = "intfloat/multilingual-e5-large"
DEFAULT_TARGET = PROJECT_ROOT / "models" / "intfloat-multilingual-e5-large"
IGNORE_PATTERNS = [
    "*.bin",
    "onnx/*",
    "openvino/*",
    "*.h5",
    "*.ot",
    "*.msgpack",
]


def download_model(target: Path = DEFAULT_TARGET) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DEFAULT_REPO,
        local_dir=target,
        ignore_patterns=IGNORE_PATTERNS,
        max_workers=4,
    )
    weights = target / "model.safetensors"
    if not weights.exists() or weights.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"Model tải chưa đầy đủ: thiếu weights hợp lệ tại {weights}")
    return target


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tải E5 embedding model về local")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    path = download_model(args.target)
    print(f"Model sẵn sàng tại: {path.resolve()}")


if __name__ == "__main__":
    main()
