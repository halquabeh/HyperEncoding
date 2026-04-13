import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def save_split(dataset_name: str, split: str, out_dir: Path) -> None:
    print(f"[{split}] Loading dataset metadata...", flush=True)
    ds = load_dataset(dataset_name, split=split)
    label_feature = ds.features["label"]
    label_names = list(label_feature.names)
    print(f"[{split}] Loaded {len(ds)} samples across {len(label_names)} classes.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, _label_name in enumerate(label_names):
        class_dir = out_dir / f"{idx:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{split}] Created class directories under {out_dir}.", flush=True)

    for sample_idx, sample in enumerate(tqdm(ds, total=len(ds), desc=f"[{split}] Exporting")):
        image = sample["image"]
        label = int(sample["label"])
        class_dir = out_dir / f"{label:03d}"
        image_path = class_dir / f"{sample_idx:06d}.png"
        if not image_path.exists():
            image.convert("RGB").save(image_path)

    label_map = {f"{idx:03d}": name for idx, name in enumerate(label_names)}
    with (out_dir.parent / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)
    print(f"[{split}] Wrote label map and finished export.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="basavyr/imagenet-100",
        help="Hugging Face dataset ID for ImageNet-100",
    )
    parser.add_argument(
        "--output",
        default="data/Image100",
        help="Output root directory",
    )
    args = parser.parse_args()

    output_root = Path(args.output)
    save_split(args.dataset, "train", output_root / "train.X")
    save_split(args.dataset, "validation", output_root / "val.X")


if __name__ == "__main__":
    main()
