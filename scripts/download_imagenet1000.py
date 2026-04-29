import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import Image, load_dataset
from tqdm import tqdm


def save_split(dataset_name: str, split: str, out_dir: Path, token: str | None = None) -> None:
    print(f"[{split}] Loading dataset metadata...", flush=True)
    try:
        ds = load_dataset(dataset_name, split=split, token=token)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {dataset_name}:{split}. "
            "For gated datasets like ImageNet-1k, accept the dataset terms on Hugging Face "
            "and authenticate with `huggingface-cli login` or pass `--token` / `HF_TOKEN`."
        ) from exc

    label_feature = ds.features["label"]
    label_names = list(label_feature.names)
    class_width = max(3, len(str(len(label_names) - 1)))
    sample_width = max(6, len(str(len(ds) - 1)))
    print(f"[{split}] Loaded {len(ds)} samples across {len(label_names)} classes.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, _label_name in enumerate(label_names):
        class_dir = out_dir / f"{idx:0{class_width}d}"
        class_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{split}] Created class directories under {out_dir}.", flush=True)

    raw_ds = ds.cast_column("image", Image(decode=False))
    for sample_idx, sample in enumerate(tqdm(raw_ds, total=len(raw_ds), desc=f"[{split}] Exporting")):
        label = int(sample["label"])
        image_info = sample["image"]
        source_path = Path(image_info["path"]) if image_info.get("path") else None
        suffix = source_path.suffix.lower() if source_path and source_path.suffix else ".jpg"
        class_dir = out_dir / f"{label:0{class_width}d}"
        image_path = class_dir / f"{sample_idx:0{sample_width}d}{suffix}"
        if image_path.exists():
            continue

        # Preserve the source encoding when possible to avoid expensive re-encoding at ImageNet scale.
        image_bytes = image_info.get("bytes")
        if image_bytes is not None:
            image_path.write_bytes(image_bytes)
        elif source_path and source_path.exists():
            shutil.copyfile(source_path, image_path)
        else:
            ds[sample_idx]["image"].convert("RGB").save(image_path)

    label_map = {f"{idx:0{class_width}d}": name for idx, name in enumerate(label_names)}
    with (out_dir.parent / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)
    print(f"[{split}] Wrote label map and finished export.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="ILSVRC/imagenet-1k",
        help="Hugging Face dataset ID for ImageNet-1k",
    )
    parser.add_argument(
        "--output",
        default="data/Image1000",
        help="Output root directory",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token. Defaults to HF_TOKEN if set.",
    )
    args = parser.parse_args()

    output_root = Path(args.output)
    save_split(args.dataset, "train", output_root / "train.X", token=args.token)
    save_split(args.dataset, "validation", output_root / "val.X", token=args.token)


if __name__ == "__main__":
    main()
