from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf


IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class DatasetBundle:
    """Lưu các tập dữ liệu và tên lớp sau khi load từ processed_dataset."""

    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]


def _validate_dataset_dirs(root: Path) -> None:
    """Kiểm tra cấu trúc processed_dataset/train, validation, test."""
    for split_name in ("train", "validation", "test"):
        split_dir = root / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {split_dir}")


def _resize_to_224x224x3(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Ép mọi ảnh đầu vào về đúng tensor RGB 224x224x3."""
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.ensure_shape(image, (None, *INPUT_SHAPE))
    return image, label


def _load_split(
    split_dir: Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tuple[tf.data.Dataset, list[str]]:
    """Load một split bằng TensorFlow và chuẩn hóa shape ảnh."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="binary",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=seed,
    )
    class_names = dataset.class_names

    # Dataset trả về batch ảnh dạng float32, shape mỗi ảnh là 224x224x3.
    dataset = dataset.map(_resize_to_224x224x3, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset, class_names


def load_processed_dataset(
    data_dir: str | Path = "processed_dataset",
    batch_size: int = 32,
    seed: int = 42,
) -> DatasetBundle:
    """Load toàn bộ processed_dataset để train model CNN."""
    root = Path(data_dir)
    _validate_dataset_dirs(root)

    train_ds, class_names = _load_split(root / "train", batch_size=batch_size, shuffle=True, seed=seed)
    validation_ds, _ = _load_split(root / "validation", batch_size=batch_size, shuffle=False, seed=seed)
    test_ds, _ = _load_split(root / "test", batch_size=batch_size, shuffle=False, seed=seed)

    return DatasetBundle(
        train=train_ds,
        validation=validation_ds,
        test=test_ds,
        class_names=class_names,
    )


if __name__ == "__main__":
    dataset_bundle = load_processed_dataset()
    image_batch, label_batch = next(iter(dataset_bundle.train))
    print(f"Các lớp: {dataset_bundle.class_names}")
    print(f"Shape batch ảnh: {image_batch.shape}")
    print(f"Shape batch nhãn: {label_batch.shape}")
