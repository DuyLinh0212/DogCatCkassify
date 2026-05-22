from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


SEED = 42
IMAGE_SIZE = (224, 224)
SPLITS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

SOURCE_DIRS = {
    "cat": Path(r"D:\DATASET\DataSetChoMeo\Cat"),
    "dog": Path(r"D:\DATASET\DataSetChoMeo\Dog"),
}

# Thư mục output mặc định nằm trong project để Loader.py đọc trực tiếp.
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "processed_dataset"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitCounts:
    train: int
    validation: int
    test: int


def collect_images(folder: Path) -> list[Path]:
    """Thu thập toàn bộ file ảnh hợp lệ trong thư mục nguồn."""
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )


def split_counts(total: int) -> SplitCounts:
    """Tính số lượng ảnh cho train, validation và test theo tỷ lệ đã cấu hình."""
    train_count = int(total * SPLITS["train"])
    validation_count = int(total * SPLITS["validation"])
    test_count = total - train_count - validation_count
    return SplitCounts(train=train_count, validation=validation_count, test=test_count)


def resize_and_save(src_path: Path, dst_path: Path) -> None:
    """Đọc ảnh, xoay đúng EXIF, đổi sang RGB và resize về 224x224."""
    with Image.open(src_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst_path, format="JPEG", quality=95, optimize=True)


def prepare_output_dirs() -> None:
    """Tạo lại cấu trúc thư mục output theo từng split và từng lớp."""
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split_name in SPLITS:
        for class_name in SOURCE_DIRS:
            (OUTPUT_ROOT / split_name / class_name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    random.seed(SEED)
    prepare_output_dirs()

    summary: dict[str, object] = {
        "image_size": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
        "seed": SEED,
        "splits": SPLITS,
        "classes": {},
        "skipped_files": [],
    }

    total_processed = 0
    total_skipped = 0

    for class_name, source_dir in SOURCE_DIRS.items():
        if not source_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {source_dir}")

        image_paths = collect_images(source_dir)
        random.shuffle(image_paths)
        counts = split_counts(len(image_paths))

        ranges = {
            "train": image_paths[: counts.train],
            "validation": image_paths[counts.train : counts.train + counts.validation],
            "test": image_paths[counts.train + counts.validation :],
        }

        class_summary = {
            "source_total": len(image_paths),
            "processed": {"train": 0, "validation": 0, "test": 0},
            "skipped": 0,
        }

        for split_name, paths in ranges.items():
            for index, src_path in enumerate(paths, start=1):
                dst_path = OUTPUT_ROOT / split_name / class_name / f"{class_name}_{index:05d}.jpg"
                try:
                    resize_and_save(src_path, dst_path)
                    class_summary["processed"][split_name] += 1
                    total_processed += 1
                except (OSError, UnidentifiedImageError) as exc:
                    class_summary["skipped"] += 1
                    total_skipped += 1
                    summary["skipped_files"].append(
                        {
                            "class": class_name,
                            "file": str(src_path),
                            "reason": str(exc),
                        }
                    )

        summary["classes"][class_name] = class_summary

    summary["total_processed"] = total_processed
    summary["total_skipped"] = total_skipped

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Tiền xử lý hoàn tất")
    print(f"Thư mục output: {OUTPUT_ROOT}")
    print(f"Tổng ảnh đã xử lý: {total_processed}")
    print(f"Tổng ảnh bị bỏ qua: {total_skipped}")
    print(f"Báo cáo: {summary_path}")


if __name__ == "__main__":
    main()
