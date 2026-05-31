from __future__ import annotations

import io
import os
import time
from pathlib import Path
import json
import sys

import tensorflow as tf
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.CNN_Model import SpatialAttention  # noqa: F401

# Khởi tạo ứng dụng FastAPI chỉ dành cho API
app = FastAPI(
    title="Sentinel Alpha Classifier API",
    description="API phân loại Chó/Mèo sử dụng các mô hình CNN đã huấn luyện (Chỉ phục vụ API)",
    version="1.0.0"
)

# Cấu hình CORS để cho phép gọi API từ máy chủ Frontend Web chạy ở cổng khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bộ nhớ đệm lưu trữ các mô hình đã tải để tránh tải lại nhiều lần
MODEL_CACHE: dict[str, tf.keras.Model] = {}
ALLOWED_MODEL_EXTENSIONS = {".keras", ".h5"}


def get_model_storage_dir() -> Path:
    """Return the directory used for models uploaded from the web UI."""
    saved_models_dir = Path(__file__).resolve().parent / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    return saved_models_dir

def find_models() -> list[dict[str, object]]:
    """Tìm tất cả các file mô hình .keras và .h5 trong thư mục models và saved_models (loại bỏ trùng lặp)."""
    model_paths = []
    seen_filenames = set()
    
    # Tìm trong thư mục models
    models_dir = Path(__file__).resolve().parent / "models"
    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ALLOWED_MODEL_EXTENSIONS:
                if f.name not in seen_filenames:
                    seen_filenames.add(f.name)
                    model_paths.append({
                        "name": f"models/{f.name}",
                        "path": str(f),
                        "filename": f.name,
                        "folder": "models",
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
                    })
                
    # Tìm trong thư mục saved_models
    saved_models_dir = get_model_storage_dir()
    if saved_models_dir.exists():
        for f in saved_models_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ALLOWED_MODEL_EXTENSIONS:
                if f.name not in seen_filenames:
                    seen_filenames.add(f.name)
                    model_paths.append({
                        "name": f"saved_models/{f.name}",
                        "path": str(f),
                        "filename": f.name,
                        "folder": "saved_models",
                        "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
                    })
                
    return model_paths

def get_model(model_name: str) -> tuple[tf.keras.Model, str]:
    """Tải động mô hình được chọn và lưu vào cache."""
    global MODEL_CACHE
    
    # Làm sạch tên mô hình tránh lỗi path traversal
    clean_name = os.path.basename(model_name)
    
    # Xác định đường dẫn thực tế của mô hình
    possible_paths = [
        Path(__file__).resolve().parent / "models" / clean_name,
        get_model_storage_dir() / clean_name
    ]
    
    selected_path = None
    for p in possible_paths:
        if p.exists() and p.is_file():
            selected_path = p
            break
            
    if not selected_path:
        raise HTTPException(
            status_code=404, 
            detail=f"Không tìm thấy mô hình '{model_name}' trên server."
        )
        
    model_path_str = str(selected_path)
    
    # Trả về mô hình từ cache nếu đã được tải trước đó
    if model_path_str in MODEL_CACHE:
        return MODEL_CACHE[model_path_str], model_path_str
        
    try:
        print(f"Bắt đầu tải mô hình: {model_path_str}")
        # compile=False giúp tải nhanh hơn và tránh lỗi do thiếu Custom Metrics như BinaryF1Score
        model = tf.keras.models.load_model(model_path_str, compile=False)
        MODEL_CACHE[model_path_str] = model
        print(f"Đã nạp thành công mô hình vào cache: {clean_name}")
        return model, model_path_str
    except Exception as e:
        print(f"Lỗi khi nạp mô hình {model_path_str}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi khi nạp mô hình lên bộ nhớ: {str(e)}"
        )

def preprocess_image(image_bytes: bytes) -> tuple[tf.Tensor, tuple[int, int]]:
    """Tiền xử lý ảnh giống hệt quy trình lúc huấn luyện mô hình."""
    # Đọc ảnh từ bytes
    img = Image.open(io.BytesIO(image_bytes))
    original_size = img.size # (width, height)
    
    # Tự động xoay ảnh đúng chiều theo dữ liệu EXIF
    img = ImageOps.exif_transpose(img)
    
    # Chuyển đổi định dạng màu sang RGB (bỏ kênh alpha nếu có)
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Resize và crop ảnh về đúng 224x224 (sử dụng Lanczos cho chất lượng cao nhất)
    img_resized = ImageOps.fit(img, (224, 224), method=Image.Resampling.LANCZOS)
    
    # Chuyển đổi sang numpy array shape (224, 224, 3)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    
    # Thêm chiều batch để thành shape (1, 224, 224, 3)
    img_batch = tf.expand_dims(img_array, 0)
    
    return img_batch, original_size

# API endpoint: Lấy danh sách các model khả dụng
@app.get("/api/models")
async def list_models():
    try:
        available_models = find_models()
        return {
            "success": True,
            "models": available_models
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )

# API endpoint: Dự đoán ảnh đơn lẻ
@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    try:
        filename = os.path.basename(file.filename or "")
        suffix = Path(filename).suffix.lower()

        if suffix not in ALLOWED_MODEL_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Chỉ hỗ trợ file model .keras hoặc .h5"
            )

        if not filename:
            raise HTTPException(status_code=400, detail="Tên file model không hợp lệ")

        storage_dir = get_model_storage_dir()
        save_path = storage_dir / filename
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="File model rỗng")

        save_path.write_bytes(contents)
        model_name = f"saved_models/{filename}"

        return {
            "success": True,
            "model": {
                "name": model_name,
                "path": str(save_path),
                "filename": filename,
                "folder": "saved_models",
                "size_mb": round(save_path.stat().st_size / (1024 * 1024), 2),
            },
            "models": find_models(),
        }
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("models/best_model.keras"),
    threshold: float = Form(0.5)
):
    try:
        contents = await file.read()
        
        # Tiền xử lý ảnh
        img_batch, original_size = preprocess_image(contents)
        
        # Lấy mô hình từ cache/tệp tin
        model, actual_path = get_model(model_name)
        
        # Chạy suy luận và đo độ trễ
        start_time = time.time()
        prediction = model.predict(img_batch)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        # Kết quả trả về của sigmoid là xác suất thuộc lớp 1 (Chó)
        raw_score = float(prediction[0][0])
        
        # Phân loại dựa trên ngưỡng (threshold) động
        if raw_score >= threshold:
            label = "Dog"
            label_vi = "Chó"
            confidence = raw_score
        else:
            label = "Cat"
            label_vi = "Mèo"
            confidence = 1.0 - raw_score
            
        device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
        
        return {
            "success": True,
            "filename": file.filename,
            "model_used": model_name,
            "actual_path": actual_path,
            "threshold": threshold,
            "raw_score": raw_score,
            "label": label,
            "label_vi": label_vi,
            "confidence": round(confidence * 100, 2),
            "latency_ms": latency_ms,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "device": device
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )

# API endpoint: Dự đoán hàng loạt (Batch Prediction)
@app.post("/api/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    model_name: str = Form("models/best_model.keras"),
    threshold: float = Form(0.5)
):
    try:
        model, actual_path = get_model(model_name)
        device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
        results = []
        
        for file in files:
            try:
                contents = await file.read()
                img_batch, original_size = preprocess_image(contents)
                
                start_time = time.time()
                prediction = model.predict(img_batch)
                latency_ms = round((time.time() - start_time) * 1000, 2)
                
                raw_score = float(prediction[0][0])
                
                if raw_score >= threshold:
                    label = "Dog"
                    label_vi = "Chó"
                    confidence = raw_score
                else:
                    label = "Cat"
                    label_vi = "Mèo"
                    confidence = 1.0 - raw_score
                    
                results.append({
                    "success": True,
                    "filename": file.filename,
                    "raw_score": raw_score,
                    "label": label,
                    "label_vi": label_vi,
                    "confidence": round(confidence * 100, 2),
                    "latency_ms": latency_ms,
                    "original_width": original_size[0],
                    "original_height": original_size[1]
                })
            except Exception as exc:
                results.append({
                    "success": False,
                    "filename": file.filename,
                    "detail": str(exc)
                })
                
        return {
            "success": True,
            "model_used": model_name,
            "actual_path": actual_path,
            "threshold": threshold,
            "device": device,
            "results": results
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": str(e)}
        )

if __name__ == "__main__":
    print("Khởi động Backend API Server (Sentinel Alpha)...")
    models_found = find_models()
    print(f"Tìm thấy {len(models_found)} mô hình khả dụng:")
    for m in models_found:
        print(f" - {m['name']} ({m['size_mb']} MB)")
        
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
