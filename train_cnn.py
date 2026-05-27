from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


# Cho phép import từ thư mục gốc project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from loader.Loader import load_processed_dataset
from models.CNN_Model import build_cnn_model, create_data_parallel_strategy


class BinaryF1Score(tf.keras.metrics.Metric):
    """
    F1-score cho bài toán phân loại nhị phân chó/mèo.
    """

    def __init__(self, name="f1_score", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) >= self.threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        return (2.0 * self.tp) / (
            2.0 * self.tp + self.fp + self.fn + tf.keras.backend.epsilon()
        )

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def reset_states(self):
        self.reset_state()


def create_optimizer(optimizer_name: str, learning_rate: float):
    """
    Chọn optimizer Adam hoặc SGD.
    """

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    if optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9
        )

    raise ValueError("Chỉ hỗ trợ optimizer: adam hoặc sgd")


def compile_cnn_model(optimizer_name: str, learning_rate: float):
    """
    Build CNN model và compile để train.
    """

    model = build_cnn_model(
        input_shape=(224, 224, 3),
        num_classes=1
    )

    optimizer = create_optimizer(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            BinaryF1Score(name="f1_score"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def save_history_json(history, output_dir: Path):
    """
    Lưu history training ra file JSON.
    """

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    history_path = output_dir / "history.json"
    history_path.write_text(
        json.dumps(history_dict, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Đã lưu history JSON: {history_path}")


def save_history_csv(history, output_dir: Path):
    """
    Lưu history training ra file CSV để đưa vào báo cáo.
    """

    csv_path = output_dir / "history.csv"
    keys = list(history.history.keys())
    epochs = len(history.history[keys[0]])

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch"] + keys)

        for epoch in range(epochs):
            row = [epoch + 1]
            for key in keys:
                row.append(history.history[key][epoch])
            writer.writerow(row)

    print(f"Đã lưu history CSV: {csv_path}")


def plot_metric(history, train_key: str, val_key: str, title: str, ylabel: str, save_path: Path):
    """
    Vẽ biểu đồ 1 metric train/validation.
    """

    if train_key not in history.history or val_key not in history.history:
        print(f"Bỏ qua biểu đồ {title} vì không tìm thấy {train_key} hoặc {val_key}")
        return

    epochs = range(1, len(history.history[train_key]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history[train_key], marker="o", label=f"Train {ylabel}")
    plt.plot(epochs, history.history[val_key], marker="o", label=f"Validation {ylabel}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Đã lưu biểu đồ: {save_path}")


def plot_all_charts(history, output_dir: Path):
    """
    Xuất các biểu đồ cần cho báo cáo.
    """

    plot_metric(
        history,
        train_key="accuracy",
        val_key="val_accuracy",
        title="Training Accuracy",
        ylabel="Accuracy",
        save_path=output_dir / "accuracy_chart.png"
    )

    plot_metric(
        history,
        train_key="loss",
        val_key="val_loss",
        title="Training Loss",
        ylabel="Loss",
        save_path=output_dir / "loss_chart.png"
    )

    plot_metric(
        history,
        train_key="f1_score",
        val_key="val_f1_score",
        title="Training F1-score",
        ylabel="F1-score",
        save_path=output_dir / "f1_score_chart.png"
    )

    plot_metric(
        history,
        train_key="precision",
        val_key="val_precision",
        title="Training Precision",
        ylabel="Precision",
        save_path=output_dir / "precision_chart.png"
    )

    plot_metric(
        history,
        train_key="recall",
        val_key="val_recall",
        title="Training Recall",
        ylabel="Recall",
        save_path=output_dir / "recall_chart.png"
    )

    plot_metric(
        history,
        train_key="auc",
        val_key="val_auc",
        title="Training AUC",
        ylabel="AUC",
        save_path=output_dir / "auc_chart.png"
    )


def main():
    parser = argparse.ArgumentParser(description="Train CNN phân loại chó/mèo")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Đường dẫn dataset có train/validation/test"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Thư mục lưu model, biểu đồ, kết quả"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Số vòng lặp training"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Chọn optimizer: adam hoặc sgd"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== CẤU HÌNH TRAIN ==========")
    print(f"Dataset       : {data_dir}")
    print(f"Output        : {output_dir}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Optimizer     : {args.optimizer}")
    print(f"Learning rate : {args.learning_rate}")
    print("====================================")

    dataset_bundle = load_processed_dataset(
        data_dir=data_dir,
        batch_size=args.batch_size
    )

    print("Class names:", dataset_bundle.class_names)

    # Lưu class_names để Tkinter dùng sau này
    class_names_path = output_dir / "class_names.json"
    class_names_path.write_text(
        json.dumps(dataset_bundle.class_names, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    strategy = create_data_parallel_strategy()

    print(f"Số GPU/replica đang dùng: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = compile_cnn_model(
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate
        )

    model.summary()

    best_model_path = output_dir / "best_model.keras"
    final_model_path = output_dir / "final_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_f1_score",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    history = model.fit(
        dataset_bundle.train,
        validation_data=dataset_bundle.validation,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print("========== ĐÁNH GIÁ TEST ==========")

    test_results = model.evaluate(
        dataset_bundle.test,
        return_dict=True
    )

    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value:.4f}")

    model.save(final_model_path)

    save_history_json(history, output_dir)
    save_history_csv(history, output_dir)
    plot_all_charts(history, output_dir)

    result = {
        "class_names": dataset_bundle.class_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "test_results": {
            key: float(value)
            for key, value in test_results.items()
        },
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "class_names_path": str(class_names_path),
    }

    result_path = output_dir / "train_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("========== TRAIN XONG ==========")
    print(f"Best model      : {best_model_path}")
    print(f"Final model     : {final_model_path}")
    print(f"Class names     : {class_names_path}")
    print(f"History JSON    : {output_dir / 'history.json'}")
    print(f"History CSV     : {output_dir / 'history.csv'}")
    print(f"Accuracy chart  : {output_dir / 'accuracy_chart.png'}")
    print(f"Loss chart      : {output_dir / 'loss_chart.png'}")
    print(f"F1 chart        : {output_dir / 'f1_score_chart.png'}")
    print(f"Train result    : {result_path}")


if __name__ == "__main__":
    main()