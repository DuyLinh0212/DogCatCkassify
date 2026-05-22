<<<<<<< HEAD
# CNN Dog Cat Classify

Dự án xây dựng mô hình CNN để phân loại ảnh chó và mèo bằng TensorFlow/Keras. Dữ liệu đầu vào được tổ chức theo cấu trúc `processed_dataset`, sau đó được load thành `tf.data.Dataset` với kích thước ảnh chuẩn `224x224x3`.

## Cấu trúc thư mục

```text
CNN_Dog_Cat_Classify/
+-- loader/
|   +-- Loader.py
+-- models/
|   +-- CNN_Model.py
+-- processed_dataset/
|   +-- train/
|   +-- validation/
|   +-- test/
|   +-- summary.json
+-- train/
+-- evaluation/
+-- .gitignore
+-- README.md
```

## Chức năng chính

- `models/CNN_Model.py`: xây dựng và compile mô hình CNN input `224x224x3`.
- `loader/Loader.py`: load dữ liệu từ `processed_dataset`, resize ảnh về `224x224x3`, batch và prefetch dữ liệu.
- `processed_dataset/`: chứa dữ liệu đã chia thành `train`, `validation`, `test`.
- `preprocessing/`: chứa code tiền xử lý dữ liệu và đang được bỏ qua bởi Git.

## Cấu trúc dữ liệu

Thư mục dữ liệu cần có dạng:

```text
processed_dataset/
+-- train/
|   +-- cat/
|   +-- dog/
+-- validation/
|   +-- cat/
|   +-- dog/
+-- test/
    +-- cat/
    +-- dog/
```

## Cách dùng

```python
from loader.Loader import load_processed_dataset
from models.CNN_Model import compile_model_data_parallel

dataset_bundle = load_processed_dataset(
    data_dir="processed_dataset",
    batch_size=32,
)

model, strategy = compile_model_data_parallel(learning_rate=1e-4)

history = model.fit(
    dataset_bundle.train,
    validation_data=dataset_bundle.validation,
    epochs=20,
)

model.evaluate(dataset_bundle.test)
```

## Chạy trên Kaggle

Khi chạy trên Kaggle, truyền đường dẫn dataset theo vị trí Kaggle mount:

```python
dataset_bundle = load_processed_dataset(
    data_dir="/kaggle/input/ten-dataset/processed_dataset",
    batch_size=32,
)
```

Hàm `compile_model_data_parallel()` dùng `tf.distribute.MirroredStrategy` khi Kaggle có nhiều GPU, giúp train data parallel tự động.

## Ghi chú Git

File `.gitignore` hiện chỉ bỏ qua thư mục:

```text
preprocessing/
```

Nếu không muốn upload dataset lớn lên Git, cần thêm `processed_dataset/` vào `.gitignore`.
=======
# AnimalClasifyByCNN
Phân loại ảnh động vật (Chó, Mèo) bằng CNN
>>>>>>> c56355015d645c1924775a749fc28743646d1cc2
