# CNN Dog Cat Classify

Dự án huấn luyện mô hình CNN để phân loại ảnh chó/mèo bằng TensorFlow/Keras. Pipeline hiện tại gồm tiền xử lý dữ liệu, load dataset theo chuẩn `tf.data`, huấn luyện CNN có lớp Attention, xuất báo cáo đánh giá và chạy giao diện web để dự đoán ảnh.

## Cấu Trúc Project

```text
CNN_Dog_Cat_Classify/
+-- application/
|   +-- api.py                  # FastAPI backend dự đoán ảnh
|   +-- web_server.py           # Static web server cho giao diện
|   +-- web/
|       +-- index.html
|       +-- css/style.css
|       +-- js/app.js
+-- loader/
|   +-- Loader.py               # Load train/validation/test bằng tf.data
+-- models/
|   +-- CNN_Model.py            # Kiến trúc CNN + SpatialAttention
+-- preprocessing/
|   +-- preprocess_dataset.py   # Tiền xử lý/chia dữ liệu
+-- train/
|   +-- train_cnn.py            # Huấn luyện và đánh giá mô hình
+-- processed_dataset/          # Dataset sau xử lý, không commit
+-- saved_models/               # Model sau train, không commit
+-- evaluation/                 # Biểu đồ và chỉ số đánh giá
+-- prediction_data/            # Dữ liệu dự đoán thử, không commit
+-- .gitignore
+-- README.md
```

## Chức Năng Chính

- `models/CNN_Model.py`: xây dựng CNN input `224x224x3`, có augmentation nhẹ, `SpatialAttention`, `GlobalAveragePooling2D` và Dense classifier.
- `loader/Loader.py`: load dataset từ thư mục có `train`, `validation`, `test`, resize ảnh về `224x224`, batch và prefetch.
- `train/train_cnn.py`: train model, hỗ trợ Adam/SGD, data parallel bằng `MirroredStrategy`, early stopping theo `val_f1_score`, lưu best/final model và toàn bộ kết quả đánh giá.
- `application/api.py`: backend FastAPI cho dự đoán ảnh đơn/batch, load model `.keras/.h5`, hỗ trợ upload model từ giao diện web.
- `application/web/`: giao diện web chọn model, import model từ máy, upload ảnh đơn hoặc nhiều ảnh để phân loại.

## Cấu Trúc Dataset

Dataset cần có dạng:

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

Nếu dùng Kaggle và dataset có `train`, `validation`, `test` nằm trực tiếp trong thư mục input, truyền chính thư mục đó vào `--data_dir`.

## Huấn Luyện

Chạy local:

```powershell
py train\train_cnn.py `
  --data_dir processed_dataset `
  --output_dir saved_models `
  --evaluation_dir evaluation `
  --epochs 50 `
  --batch_size 32 `
  --optimizer adam `
  --learning_rate 0.0001 `
  --early_stopping_patience 10
```

Chạy trên Kaggle:

```python
!python train/train_cnn.py \
  --data_dir /kaggle/input/dogcatdata \
  --output_dir /kaggle/working/saved_models \
  --evaluation_dir /kaggle/working/evaluation \
  --epochs 50 \
  --batch_size 32 \
  --optimizer adam \
  --learning_rate 0.0001 \
  --early_stopping_patience 10
```

Các tham số quan trọng:

```text
--data_dir                  Đường dẫn dataset có train/validation/test
--output_dir                Nơi lưu best_model.keras và final_model.keras
--evaluation_dir            Nơi lưu biểu đồ, metrics, confusion matrix
--epochs                    Số epoch tối đa
--batch_size                Batch size
--optimizer                 adam hoặc sgd
--learning_rate             Learning rate
--early_stopping_patience   Số epoch chờ theo val_f1_score
--show_model_summary        In model.summary() nếu cần
```

## Kết Quả Sau Train

Trong `saved_models/`:

```text
best_model.keras     # Model tốt nhất theo val_f1_score, nên dùng để demo/dự đoán
final_model.keras    # Model ở cuối quá trình train
class_names.json
```

Trong `evaluation/`:

```text
history.json
history.csv
accuracy_chart.png
loss_chart.png
f1_score_chart.png
precision_chart.png
recall_chart.png
auc_chart.png
test_metrics.json
test_metrics.csv
classification_metrics.json
classification_metrics.csv
confusion_matrix.png
confusion_matrix.csv
model_evaluation_metrics.png
train_result.json
```

## Chạy Ứng Dụng Web

Backend API cần TensorFlow, FastAPI, Pillow và Uvicorn. Nên dùng virtual environment riêng, ưu tiên Python 3.12 nếu TensorFlow trên Python 3.13 bị lỗi.

```powershell
cd F:\NgDuyLinh\Do_an\ThucHangDeep\CNN_Dog_Cat_Classify
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install tensorflow fastapi uvicorn pillow python-multipart numpy matplotlib
```

Chạy API:

```powershell
cd application
python api.py
```

API chạy tại:

```text
http://127.0.0.1:8000
```

Chạy frontend ở terminal khác:

```powershell
cd F:\NgDuyLinh\Do_an\ThucHangDeep\CNN_Dog_Cat_Classify\application
py web_server.py
```

Mở web:

```text
http://localhost:3000
```

Trong giao diện web có thể:

- Nhập model `.keras` hoặc `.h5` từ máy.
- Chọn model đã upload.
- Upload một ảnh để dự đoán.
- Upload nhiều ảnh để dự đoán batch.
- Xuất lịch sử dự đoán dạng CSV.

## Ghi Chú

- Dùng `best_model.keras` cho demo và dự đoán thực tế.
- `processed_dataset/`, `prediction_data/`, `saved_models/`, `__pycache__/`, `*.pyc`, `.vscode/` đang được bỏ qua bởi Git.
- Nếu model có custom layer `SpatialAttention`, cần import `models.CNN_Model` trước khi `load_model`; `application/api.py` đã xử lý việc này.
