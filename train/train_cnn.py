from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass


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


def _format_metric(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_learning_rate(optimizer):
    learning_rate = optimizer.learning_rate
    if callable(learning_rate):
        learning_rate = learning_rate(optimizer.iterations)
    return _format_metric(tf.keras.backend.get_value(learning_rate))


class CompactProgressLogger(tf.keras.callbacks.Callback):
    """
    Print a compact progress bar with only the most useful training metrics.
    """

    def __init__(
        self,
        bar_width: int = 16,
        monitor: str = "val_f1_score",
        best_model_path: Path | None = None,
    ):
        super().__init__()
        self.bar_width = bar_width
        self.monitor = monitor
        self.best_model_path = best_model_path
        self.best_value = float("-inf")
        self.total_epochs = 0
        self.steps = None
        self.epoch_start_time = 0.0
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get("epochs", 0)
        self.steps = self.params.get("steps")
        print("\nTraining progress")

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self._render(epoch, 0, {})

    def on_train_batch_end(self, batch, logs=None):
        self._render(self.current_epoch, batch + 1, logs or {})

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = time.time() - self.epoch_start_time
        best_message = ""
        current_value = logs.get(self.monitor)

        if current_value is not None and float(current_value) > self.best_value:
            self.best_value = float(current_value)
            best_message = " - saved best"

        summary = (
            f" | val_loss={_format_metric(logs.get('val_loss'))}"
            f" val_acc={_format_metric(logs.get('val_accuracy'))}"
            f" val_f1={_format_metric(logs.get('val_f1_score'))}"
            f" - {elapsed:.1f}s"
            f"{best_message}"
        )
        self._render(epoch, self.steps, logs)
        print(summary)

    def _render(self, epoch, step, logs):
        if self.steps:
            progress = min(step / self.steps, 1.0)
            filled = int(self.bar_width * progress)
            bar = "=" * filled + "." * (self.bar_width - filled)
            step_text = f"{step}/{self.steps}"
        else:
            bar = "." * self.bar_width
            step_text = str(step or 0)

        metrics = (
            f"loss={_format_metric(logs.get('loss'))}"
            f" acc={_format_metric(logs.get('accuracy'))}"
            f" f1={_format_metric(logs.get('f1_score'))}"
        )
        line = (
            f"\rEpoch {epoch + 1}/{self.total_epochs} "
            f"[{bar}] {step_text} | {metrics}"
        )
        print(line, end="", flush=True)


def print_key_value_table(title: str, values: dict):
    key_width = max([len(str(key)) for key in values.keys()] + [6])
    value_width = max([len(str(value)) for value in values.values()] + [5])
    separator = f"+-{'-' * key_width}-+-{'-' * value_width}-+"

    print(f"\n{title}")
    print(separator)
    print(f"| {'Metric':<{key_width}} | {'Value':>{value_width}} |")
    print(separator)
    for key, value in values.items():
        print(f"| {str(key):<{key_width}} | {str(value):>{value_width}} |")
    print(separator)


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


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def save_test_metrics(test_results: dict[str, float], output_dir: Path):
    """
    Save model.evaluate() metrics as JSON and CSV.
    """

    metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in test_results.items()
    }

    json_path = output_dir / "test_metrics.json"
    csv_path = output_dir / "test_metrics.csv"

    json_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for metric_name, metric_value in metrics.items():
            writer.writerow([metric_name, metric_value])

    print(f"Saved test metrics JSON: {json_path}")
    print(f"Saved test metrics CSV: {csv_path}")


def collect_binary_predictions(model: tf.keras.Model, dataset: tf.data.Dataset):
    """
    Collect y_true, probabilities, and binary predictions from a labeled dataset.
    """

    y_true_batches = []
    image_dataset = dataset.map(
        lambda images, labels: images,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    for _, labels in dataset:
        y_true_batches.append(labels.numpy().reshape(-1))

    y_true = np.concatenate(y_true_batches).astype(int)
    y_prob = model.predict(image_dataset, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    return y_true, y_prob, y_pred


def create_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Create a 2x2 confusion matrix for binary classification.
    """

    return tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=2
    ).numpy()


def save_confusion_matrix_csv(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    save_path: Path,
):
    """
    Save confusion matrix as CSV.
    """

    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label/predicted_label"] + class_names)

        for class_name, row in zip(class_names, confusion_matrix):
            writer.writerow([class_name] + [int(value) for value in row])

    print(f"Saved confusion matrix CSV: {save_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    save_path: Path,
):
    """
    Save confusion matrix heatmap.
    """

    plt.figure(figsize=(6, 5))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = confusion_matrix.max() / 2.0 if confusion_matrix.size else 0
    for row_index in range(confusion_matrix.shape[0]):
        for col_index in range(confusion_matrix.shape[1]):
            value = int(confusion_matrix[row_index, col_index])
            text_color = "white" if value > threshold else "black"
            plt.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved confusion matrix chart: {save_path}")


def build_classification_metrics(
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    Calculate per-class, macro, weighted, and overall metrics from a confusion matrix.
    """

    total = int(confusion_matrix.sum())
    correct = int(np.trace(confusion_matrix))
    per_class = {}

    for class_index, class_name in enumerate(class_names):
        tp = int(confusion_matrix[class_index, class_index])
        fp = int(confusion_matrix[:, class_index].sum() - tp)
        fn = int(confusion_matrix[class_index, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        support = int(confusion_matrix[class_index, :].sum())

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1_score = _safe_divide(2.0 * precision * recall, precision + recall)

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    macro_precision = float(np.mean([item["precision"] for item in per_class.values()]))
    macro_recall = float(np.mean([item["recall"] for item in per_class.values()]))
    macro_f1 = float(np.mean([item["f1_score"] for item in per_class.values()]))

    weighted_precision = _safe_divide(
        sum(item["precision"] * item["support"] for item in per_class.values()),
        total
    )
    weighted_recall = _safe_divide(
        sum(item["recall"] * item["support"] for item in per_class.values()),
        total
    )
    weighted_f1 = _safe_divide(
        sum(item["f1_score"] * item["support"] for item in per_class.values()),
        total
    )

    return {
        "overall": {
            "accuracy": _safe_divide(correct, total),
            "total_samples": total,
            "correct_samples": correct,
            "incorrect_samples": total - correct,
        },
        "per_class": per_class,
        "macro_avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
        },
        "weighted_avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1,
        },
    }


def save_classification_metrics(metrics: dict, output_dir: Path):
    """
    Save detailed classification metrics as JSON and CSV.
    """

    json_path = output_dir / "classification_metrics.json"
    csv_path = output_dir / "classification_metrics.csv"

    json_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "class",
            "precision",
            "recall",
            "f1_score",
            "support",
            "tp",
            "fp",
            "fn",
            "tn",
        ])

        for class_name, values in metrics["per_class"].items():
            writer.writerow([
                class_name,
                values["precision"],
                values["recall"],
                values["f1_score"],
                values["support"],
                values["tp"],
                values["fp"],
                values["fn"],
                values["tn"],
            ])

    print(f"Saved classification metrics JSON: {json_path}")
    print(f"Saved classification metrics CSV: {csv_path}")


def plot_test_metrics(test_results: dict[str, float], save_path: Path):
    """
    Save a bar chart for scalar test metrics.
    """

    metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in test_results.items()
        if np.isscalar(metric_value)
    }

    if not metrics:
        print("Skip test metrics chart because no valid metrics were found")
        return

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    plt.figure(figsize=(9, 5))
    bars = plt.bar(metric_names, metric_values, color="#2f6f8f")
    plt.title("Test Evaluation Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved test metrics chart: {save_path}")


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
        help="Thư mục lưu model"
    )

    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="evaluation",
        help="Thư mục lưu biểu đồ, ma trận nhầm lẫn và chỉ số đánh giá"
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

    parser.add_argument(
        "--show_model_summary",
        action="store_true",
        help="In model.summary() neu can xem chi tiet kien truc"
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="So epoch cho EarlyStopping theo val_f1_score"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    evaluation_dir = Path(args.evaluation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    print_key_value_table(
        "TRAIN CONFIG",
        {
            "Dataset": data_dir,
            "Output": output_dir,
            "Evaluation": evaluation_dir,
            "Epochs": args.epochs,
            "Batch size": args.batch_size,
            "Optimizer": args.optimizer,
            "Learning rate": args.learning_rate,
            "Early stopping": f"val_f1_score patience={args.early_stopping_patience}",
        },
    )

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
    evaluation_class_names_path = evaluation_dir / "class_names.json"
    evaluation_class_names_path.write_text(
        json.dumps(dataset_bundle.class_names, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    strategy = create_data_parallel_strategy()

    print_key_value_table(
        "RUNTIME",
        {
            "Replicas": strategy.num_replicas_in_sync,
            "Classes": ", ".join(dataset_bundle.class_names),
        },
    )

    with strategy.scope():
        model = compile_cnn_model(
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate
        )

    if args.show_model_summary:
        model.summary()

    best_model_path = output_dir / "best_model.keras"
    final_model_path = output_dir / "final_model.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_f1_score",
            save_best_only=True,
            mode="max",
            verbose=0
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score",
            mode="max",
            patience=args.early_stopping_patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=0
        ),

        CompactProgressLogger(best_model_path=best_model_path),
    ]

    history = model.fit(
        dataset_bundle.train,
        validation_data=dataset_bundle.validation,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0
    )

    print("\nTEST EVALUATION")

    test_results = model.evaluate(
        dataset_bundle.test,
        return_dict=True,
        verbose=0
    )

    print_key_value_table(
        "TEST METRICS",
        {
            metric_name: _format_metric(metric_value)
            for metric_name, metric_value in test_results.items()
        },
    )

    model.save(final_model_path)

    evaluation_class_names = dataset_bundle.class_names
    if len(evaluation_class_names) != 2:
        evaluation_class_names = ["class_0", "class_1"]

    save_history_json(history, evaluation_dir)
    save_history_csv(history, evaluation_dir)
    plot_all_charts(history, evaluation_dir)
    save_test_metrics(test_results, evaluation_dir)
    plot_test_metrics(
        test_results,
        save_path=evaluation_dir / "model_evaluation_metrics.png"
    )

    y_true, y_prob, y_pred = collect_binary_predictions(
        model=model,
        dataset=dataset_bundle.test
    )
    confusion_matrix = create_confusion_matrix(y_true, y_pred)
    save_confusion_matrix_csv(
        confusion_matrix=confusion_matrix,
        class_names=evaluation_class_names,
        save_path=evaluation_dir / "confusion_matrix.csv"
    )
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=evaluation_class_names,
        save_path=evaluation_dir / "confusion_matrix.png"
    )
    classification_metrics = build_classification_metrics(
        confusion_matrix=confusion_matrix,
        class_names=evaluation_class_names
    )
    save_classification_metrics(classification_metrics, evaluation_dir)

    result = {
        "class_names": dataset_bundle.class_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "early_stopping_monitor": "val_f1_score",
        "early_stopping_patience": args.early_stopping_patience,
        "test_results": {
            key: float(value)
            for key, value in test_results.items()
        },
        "classification_metrics": classification_metrics,
        "confusion_matrix": confusion_matrix.astype(int).tolist(),
        "test_prediction_threshold": 0.5,
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "class_names_path": str(class_names_path),
        "evaluation_class_names_path": str(evaluation_class_names_path),
        "evaluation_dir": str(evaluation_dir),
    }

    result_path = evaluation_dir / "train_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("========== TRAIN XONG ==========")
    print(f"Best model      : {best_model_path}")
    print(f"Final model     : {final_model_path}")
    print(f"Class names     : {class_names_path}")
    print(f"Evaluation dir  : {evaluation_dir}")
    print(f"History JSON    : {evaluation_dir / 'history.json'}")
    print(f"History CSV     : {evaluation_dir / 'history.csv'}")
    print(f"Accuracy chart  : {evaluation_dir / 'accuracy_chart.png'}")
    print(f"Loss chart      : {evaluation_dir / 'loss_chart.png'}")
    print(f"F1 chart        : {evaluation_dir / 'f1_score_chart.png'}")
    print(f"Test metrics    : {evaluation_dir / 'test_metrics.json'}")
    print(f"Confusion matrix: {evaluation_dir / 'confusion_matrix.png'}")
    print(f"Train result    : {result_path}")


if __name__ == "__main__":
    main()
