from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


INPUT_SHAPE = (224, 224, 3)


@tf.keras.utils.register_keras_serializable(package="CNNDogCat")
class SpatialAttention(layers.Layer):
    """Lightweight spatial attention block for CNN feature maps."""

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-padding attention.")

        self.kernel_size = kernel_size
        self.attention_conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            use_bias=False,
        )

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        attention_map = self.attention_conv(tf.concat([avg_pool, max_pool], axis=-1))
        return inputs * attention_map

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def create_data_parallel_strategy() -> tf.distribute.Strategy:
    """Tạo chiến lược data parallel cho Kaggle hoặc máy có nhiều GPU."""
    gpus = tf.config.list_physical_devices("GPU")

    # Bật cấp phát bộ nhớ linh hoạt để TensorFlow không chiếm toàn bộ VRAM ngay từ đầu.
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Nếu TensorFlow đã khởi tạo GPU trước đó thì không thể đổi memory growth nữa.
            pass

    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy()

    return tf.distribute.get_strategy()


def build_cnn_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = 1,
) -> tf.keras.Model:
    """Xây dựng mô hình CNN phân loại chó/mèo với input 224x224x3."""
    if input_shape != INPUT_SHAPE:
        raise ValueError("Mô hình yêu cầu input đúng kích thước 224x224x3.")

    output_units = 1 if num_classes == 1 else num_classes
    output_activation = "sigmoid" if output_units == 1 else "softmax"

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            # Augmentation nhẹ, chỉ hoạt động trong lúc training.
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),

            # Chuẩn hóa pixel từ [0, 255] về [0, 1] ngay trong model.
            layers.Rescaling(1.0 / 255),

            # Khối tích chập 1: học các đặc trưng cạnh, màu và texture đơn giản.
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.20),

            # Khối tích chập 2: tăng số filter để học đặc trưng phức tạp hơn.
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Khối tích chập 3: nhận diện hình dạng lớn hơn của chó/mèo.
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.30),

            # Khối tích chập 4: gom đặc trưng cấp cao trước khi phân loại.
            layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            SpatialAttention(name="spatial_attention"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),

            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="swish"),
            layers.BatchNormalization(),
            layers.Dropout(0.45),
            layers.Dense(output_units, activation=output_activation),
        ],
        name="cnn_cho_meo_224",
    )

    return model


def compile_model(
    learning_rate: float = 1e-4,
    num_classes: int = 1,
) -> tf.keras.Model:
    """Tạo và compile model để train trực tiếp với tf.data.Dataset."""
    model = build_cnn_model(num_classes=num_classes)
    loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
            if num_classes == 1
            else tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc")
            if num_classes == 1
            else tf.keras.metrics.SparseTopKCategoricalAccuracy(name="top_k"),
        ],
    )

    return model


def compile_model_data_parallel(
    learning_rate: float = 1e-4,
    num_classes: int = 1,
) -> tuple[tf.keras.Model, tf.distribute.Strategy]:
    """Compile model trong strategy.scope() để chạy data parallel trên nhiều GPU."""
    strategy = create_data_parallel_strategy()

    with strategy.scope():
        model = compile_model(learning_rate=learning_rate, num_classes=num_classes)

    return model, strategy


if __name__ == "__main__":
    model, strategy = compile_model_data_parallel()
    print(f"Số replica đang dùng: {strategy.num_replicas_in_sync}")
    model.summary()
