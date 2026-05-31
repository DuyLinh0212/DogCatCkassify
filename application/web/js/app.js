/* ==========================================================================
   SENTINEL ALPHA CONTROLLER - web/js/app.js
   Dynamic client logic: API bindings, Drag & Drop, Local Thresholding, History
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    
    // Global State
    let availableModels = [];
    let selectedModel = null;
    let activeSingleFile = null;
    let activePrediction = null; // Stores active single-image inference metadata
    let historyList = [];
    let batchFiles = [];

    // Elements
    const apiStatusDot = document.getElementById('api-status-dot');
    const apiStatusText = document.getElementById('api-status-text');
    const modelSelector = document.getElementById('model-selector');
    const modelUploadInput = document.getElementById('model-upload-input');
    const btnUploadModel = document.getElementById('btn-upload-model');
    const modelUploadStatus = document.getElementById('model-upload-status');
    // modelInfo elements removed
    
    // Cấu hình URL của Backend API Server riêng biệt
    const API_BASE_URL = 'http://localhost:8000';

    // Ngưỡng quyết định mặc định của mô hình là 0.5
    const DEFAULT_THRESHOLD = 0.5;
    
    // Single Analysis Elements
    const singleDropzone = document.getElementById('single-dropzone');
    const singleFileInput = document.getElementById('single-file-input');
    const singlePreviewContainer = document.getElementById('single-preview-container');
    const singlePreviewImg = document.getElementById('single-preview-img');
    const singleMetaBadge = document.getElementById('single-meta-badge');
    const btnPredict = document.getElementById('btn-predict');
    const btnClearSingle = document.getElementById('btn-clear-single');
    
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsPane = document.getElementById('results-pane');
    const metaLatency = document.getElementById('meta-latency');
    const metaRawScore = document.getElementById('meta-raw-score');
    const metaHardware = document.getElementById('meta-hardware');
    const metaResolution = document.getElementById('meta-resolution');

    // Batch Elements
    const batchDropzone = document.getElementById('batch-dropzone');
    const batchFileInput = document.getElementById('batch-file-input');
    const batchResultsCard = document.getElementById('batch-results-card');
    const batchTotalCount = document.getElementById('batch-total-count');
    const batchDogCount = document.getElementById('batch-dog-count');
    const batchCatCount = document.getElementById('batch-cat-count');
    const btnBatchPredict = document.getElementById('btn-batch-predict');
    const btnClearBatch = document.getElementById('btn-clear-batch');
    const batchProgressWrapper = document.getElementById('batch-overall-progress-wrapper');
    const batchProgressLabel = document.getElementById('batch-progress-label');
    const batchProgressBarFill = document.getElementById('batch-progress-bar-fill');
    const batchTableBody = document.getElementById('batch-table-body');

    // History Elements
    const historyContainer = document.getElementById('history-container');
    const exportCsvBtn = document.getElementById('export-csv-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // ==========================================================================
    // 1. API Initialization & Model Discovery
    // ==========================================================================
    async function checkApiConnection() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models`);
            if (response.ok) {
                const data = await response.json();
                apiStatusDot.className = 'status-indicator';
                apiStatusText.textContent = 'API Sẵn sàng';
                
                if (data.success && data.models.length > 0) {
                    availableModels = data.models;
                    populateModels(availableModels);
                } else {
                    modelSelector.innerHTML = '<option value="" disabled selected>Không tìm thấy model nào trong thư mục models/</option>';
                }
            } else {
                setApiOffline('Lỗi kết nối API');
            }
        } catch (error) {
            setApiOffline('Mất kết nối API');
        }
    }

    function setApiOffline(reason) {
        apiStatusDot.className = 'status-indicator offline';
        apiStatusText.textContent = reason;
        modelSelector.innerHTML = '<option value="" disabled selected>Không thể kết nối API</option>';
        // No panel to hide
    }

    function populateModels(models) {
        modelSelector.innerHTML = '';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = model.filename;
            modelSelector.appendChild(option);
        });
        
        // Select first model by default
        if (models.length > 0) {
            modelSelector.value = models[0].name;
            selectedModel = models[0];
        }
    }

    modelSelector.addEventListener('change', (e) => {
        selectedModel = availableModels.find(m => m.name === e.target.value);
    });

    modelUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const lowerName = file.name.toLowerCase();
        if (!lowerName.endsWith('.keras') && !lowerName.endsWith('.h5')) {
            modelUploadStatus.textContent = 'Chỉ nhận file .keras hoặc .h5';
            modelUploadStatus.className = 'model-upload-status error';
            modelUploadInput.value = '';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        btnUploadModel.classList.add('is-disabled');
        modelUploadStatus.textContent = `Đang nhập ${file.name}...`;
        modelUploadStatus.className = 'model-upload-status';

        try {
            const response = await fetch(`${API_BASE_URL}/api/models/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.detail || 'Không thể nhập model');
            }

            availableModels = data.models || [];
            populateModels(availableModels);
            modelSelector.value = data.model.name;
            selectedModel = availableModels.find(m => m.name === data.model.name) || data.model;
            modelUploadStatus.textContent = `Đã nhập ${data.model.filename}`;
            modelUploadStatus.className = 'model-upload-status success';
        } catch (error) {
            modelUploadStatus.textContent = error.message;
            modelUploadStatus.className = 'model-upload-status error';
        } finally {
            btnUploadModel.classList.remove('is-disabled');
            modelUploadInput.value = '';
        }
    });

    // ==========================================================================
    // 2. Interactive Decision Threshold Slider (Đã loại bỏ theo yêu cầu sử dụng mô hình gốc)
    // ==========================================================================

    function updateSinglePredictionUI() {
        if (!activePrediction) return;
        
        const threshold = DEFAULT_THRESHOLD;
        const score = activePrediction.raw_score;
        
        let label = "";
        let labelVi = "";
        let confidence = 0;
        
        if (score >= threshold) {
            label = "Dog";
            labelVi = "Chó";
            confidence = score; // dog confidence maps directly to raw output
        } else {
            label = "Cat";
            labelVi = "Mèo";
            confidence = 1.0 - score; // cat confidence is inverse
        }
        
        const confidencePercent = (confidence * 100).toFixed(2);
        
        // Update prediction hero card design (Sentinel Alpha styling)
        const heroCard = document.getElementById('result-hero-card');
        const classIcon = document.getElementById('result-class-icon');
        const classLabel = document.getElementById('result-class-label');
        const confVal = document.getElementById('result-confidence-val');
        
        heroCard.className = 'prediction-hero ' + label.toLowerCase();
        classIcon.textContent = label === 'Dog' ? '🐶' : '🐱';
        classLabel.textContent = labelVi;
        confVal.textContent = confidencePercent + '%';
        
        // Update Gauge visualization bar
        const barFill = document.getElementById('gauge-bar-fill');
        barFill.className = 'gauge-bar-inner ' + label.toLowerCase();
        barFill.style.width = (score * 100) + '%';
        
        document.getElementById('gauge-conf-percent').textContent = (score * 100).toFixed(1) + '%';
        metaRawScore.textContent = score.toFixed(5);
    }

    // ==========================================================================
    // 3. File Drag-and-Drop (Single Mode)
    // ==========================================================================
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        singleDropzone.addEventListener(eventName, preventDefaults, false);
        batchDropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Toggle highlight class when dragging
    ['dragenter', 'dragover'].forEach(eventName => {
        singleDropzone.addEventListener(eventName, () => singleDropzone.classList.add('dragover'), false);
        batchDropzone.addEventListener(eventName, () => batchDropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        singleDropzone.addEventListener(eventName, () => singleDropzone.classList.remove('dragover'), false);
        batchDropzone.addEventListener(eventName, () => batchDropzone.classList.remove('dragover'), false);
    });

    // Drop handler
    singleDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSingleFile(files[0]);
        }
    });

    singleDropzone.addEventListener('click', () => {
        singleFileInput.click();
    });

    singleFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSingleFile(e.target.files[0]);
        }
    });

    function handleSingleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chỉ chọn các tệp tin hình ảnh (.jpg, .png, v.v.)');
            return;
        }
        
        activeSingleFile = file;
        
        // Read file for preview
        const reader = new FileReader();
        reader.onload = (e) => {
            singlePreviewImg.src = e.target.result;
            
            // Get image dimensions dynamically
            const tempImg = new Image();
            tempImg.onload = () => {
                singleMetaBadge.textContent = `${tempImg.width} x ${tempImg.height} px | ${(file.size / 1024).toFixed(1)} KB`;
                metaResolution.textContent = `${tempImg.width} x ${tempImg.height} px`;
            };
            tempImg.src = e.target.result;
            
            // Toggle visibility of panels
            singleDropzone.style.display = 'none';
            singlePreviewContainer.style.display = 'flex';
            btnClearSingle.style.display = 'block';
            
            // Enable predict button
            btnPredict.removeAttribute('disabled');
        };
        reader.readAsDataURL(file);
    }

    btnClearSingle.addEventListener('click', () => {
        activeSingleFile = null;
        activePrediction = null;
        singleFileInput.value = '';
        
        singlePreviewImg.src = '';
        singleDropzone.style.display = 'flex';
        singlePreviewContainer.style.display = 'none';
        btnClearSingle.style.display = 'none';
        btnPredict.setAttribute('disabled', 'true');
        
        // Reset Results Panel
        resultsPlaceholder.style.display = 'flex';
        resultsPane.style.display = 'none';
    });

    // ==========================================================================
    // 4. API Request: Single Image Classification
    // ==========================================================================
    btnPredict.addEventListener('click', async () => {
        if (!activeSingleFile) return;
        
        // Show loading state
        btnPredict.setAttribute('disabled', 'true');
        const origContent = btnPredict.innerHTML;
        btnPredict.innerHTML = '<div class="progress-spinner"></div> Đang phân tích...';
        
        const formData = new FormData();
        formData.append('file', activeSingleFile);
        formData.append('model_name', modelSelector.value);
        formData.append('threshold', DEFAULT_THRESHOLD);
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    // Update active prediction
                    activePrediction = {
                        raw_score: data.raw_score,
                        filename: data.filename,
                        latency_ms: data.latency_ms,
                        device: data.device,
                        model_used: data.model_used,
                        width: data.original_width,
                        height: data.original_height
                    };
                    
                    // Show result panel
                    resultsPlaceholder.style.display = 'none';
                    resultsPane.style.display = 'flex';
                    
                    // Populate other metadata
                    metaLatency.textContent = data.latency_ms + ' ms';
                    metaHardware.textContent = data.device;
                    metaHardware.className = 'chip ' + data.device.toLowerCase();
                    
                    // Live UI update based on threshold
                    updateSinglePredictionUI();
                    
                    // Add to session history
                    addToHistory({
                        filename: data.filename,
                        thumbnail: singlePreviewImg.src,
                        label: data.label,
                        label_vi: data.label_vi,
                        confidence: data.confidence,
                        raw_score: data.raw_score,
                        latency_ms: data.latency_ms,
                        model_used: data.model_used,
                        timestamp: new Date().toLocaleTimeString()
                    });
                } else {
                    alert('Lỗi phân loại ảnh: ' + data.detail);
                }
            } else {
                alert('Lỗi kết nối máy chủ phân loại.');
            }
        } catch (error) {
            alert('Có lỗi xảy ra: ' + error.message);
        } finally {
            btnPredict.removeAttribute('disabled');
            btnPredict.innerHTML = origContent;
        }
    });

    // ==========================================================================
    // 5. Prediction History Sidebar & CSV Exporter
    // ==========================================================================
    function addToHistory(item) {
        historyList.unshift(item); // Add to beginning of history list
        if (historyList.length > 15) {
            historyList.pop(); // Keep max 15 items
        }
        
        // Save to LocalStorage for persistence across page refresh
        try {
            localStorage.setItem('sentinel_history', JSON.stringify(historyList));
        } catch (e) {
            // If quotas exceeded due to base64 images, do not fail
        }
        
        renderHistory();
    }

    function renderHistory() {
        historyContainer.innerHTML = '';
        
        if (historyList.length === 0) {
            historyContainer.innerHTML = '<div class="empty-history">Chưa có lịch sử phân tích trong phiên này</div>';
            exportCsvBtn.setAttribute('disabled', 'true');
            clearHistoryBtn.setAttribute('disabled', 'true');
            return;
        }
        
        exportCsvBtn.removeAttribute('disabled');
        clearHistoryBtn.removeAttribute('disabled');
        
        historyList.forEach((item, index) => {
            const el = document.createElement('div');
            el.className = 'history-item';
            el.innerHTML = `
                <img src="${item.thumbnail || 'placeholder.jpg'}" alt="Thumb" class="history-thumbnail">
                <div class="history-info">
                    <span class="history-file" title="${item.filename}">${item.filename}</span>
                    <span class="history-meta">${item.timestamp} | ${item.latency_ms} ms</span>
                </div>
                <div class="history-result">
                    <span class="history-badge ${item.label.toLowerCase()}">${item.label_vi}</span>
                    <span class="history-conf">${item.confidence}%</span>
                </div>
            `;
            
            // Click to reload prediction results (WOW feature)
            el.addEventListener('click', () => {
                loadHistoryItemToWorkbench(item);
            });
            
            historyContainer.appendChild(el);
        });
    }

    function loadHistoryItemToWorkbench(item) {
        // Toggle tab to single analysis if not already
        document.querySelector('.tab-btn[data-tab="single-tab"]').click();
        
        // Load image preview
        singlePreviewImg.src = item.thumbnail;
        singleMetaBadge.textContent = `Lịch sử | Model: ${item.model_used.split('/').pop()}`;
        singleDropzone.style.display = 'none';
        singlePreviewContainer.style.display = 'flex';
        btnClearSingle.style.display = 'block';
        btnPredict.removeAttribute('disabled');
        
        // Load prediction score
        activePrediction = {
            raw_score: item.raw_score,
            filename: item.filename,
            latency_ms: item.latency_ms,
            device: 'CPU', // fallback visual
            model_used: item.model_used,
            width: 0,
            height: 0
        };
        
        // Show result panel
        resultsPlaceholder.style.display = 'none';
        resultsPane.style.display = 'flex';
        
        // Populate static meta
        metaLatency.textContent = item.latency_ms + ' ms';
        metaHardware.textContent = 'CACHE';
        metaHardware.className = 'chip cpu';
        metaResolution.textContent = 'Đã lưu lịch sử';
        
        // Update visual gauges
        updateSinglePredictionUI();
    }

    clearHistoryBtn.addEventListener('click', () => {
        if (confirm('Bạn có muốn xóa toàn bộ lịch sử phân loại trong phiên này không?')) {
            historyList = [];
            localStorage.removeItem('sentinel_history');
            renderHistory();
        }
    });

    // CSV Exporter
    exportCsvBtn.addEventListener('click', () => {
        if (historyList.length === 0) return;
        
        let csvContent = '\uFEFF'; // BOM for Excel Vietnamese encoding
        csvContent += 'Thơi gian,Tên File,Model Sử dụng,Ngưỡng,Raw Score (Sigmoid),Kết quả,Độ tin cậy (%),Đô trê (ms)\n';
        
        historyList.forEach(item => {
            const row = [
                item.timestamp,
                `"${item.filename.replace(/"/g, '""')}"`,
                item.model_used,
                DEFAULT_THRESHOLD,
                item.raw_score.toFixed(5),
                item.label_vi,
                item.confidence,
                item.latency_ms
            ].join(',');
            csvContent += row + '\n';
        });
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `sentinel_alpha_report_${new Date().toISOString().slice(0, 10)}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Load history from LocalStorage if it exists
    try {
        const storedHistory = localStorage.getItem('sentinel_history');
        if (storedHistory) {
            historyList = JSON.parse(storedHistory);
            renderHistory();
        }
    } catch (e) {}

    // ==========================================================================
    // 6. Batch Processing (Multiple Image Mode)
    // ==========================================================================
    batchDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleBatchFiles(files);
        }
    });

    batchDropzone.addEventListener('click', () => {
        batchFileInput.click();
    });

    batchFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleBatchFiles(e.target.files);
        }
    });

    function handleBatchFiles(files) {
        const newFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
        if (newFiles.length === 0) return;
        
        batchFiles = [...batchFiles, ...newFiles];
        
        // Show table card
        batchResultsCard.style.display = 'block';
        batchDropzone.style.display = 'none';
        
        updateBatchCounters();
        
        // Render initial rows as "Chờ..."
        newFiles.forEach((file, index) => {
            const tr = document.createElement('tr');
            tr.id = `batch-row-${batchFiles.length - newFiles.length + index}`;
            
            // Create local object URL for preview thumbnails (Super fast local URL)
            const objectUrl = URL.createObjectURL(file);
            
            tr.innerHTML = `
                <td><img src="${objectUrl}" class="batch-thumbnail"></td>
                <td style="font-weight: 500; font-family: monospace;">${file.filename || file.name}</td>
                <td>${(file.size / 1024).toFixed(1)} KB</td>
                <td style="text-align: center;"><span class="chip" style="background-color: var(--color-surface-container);">Chờ xử lý...</span></td>
                <td style="text-align: right; font-weight: 600;">-</td>
                <td style="text-align: right; color: var(--color-on-surface-variant); font-size: 12px;">-</td>
            `;
            batchTableBody.appendChild(tr);
        });
    }

    function updateBatchCounters() {
        batchTotalCount.textContent = batchFiles.length;
        
        // Count predicted labels
        let dogs = 0;
        let cats = 0;
        
        document.querySelectorAll('.batch-class-badge').forEach(badge => {
            if (badge.classList.contains('dog')) dogs++;
            if (badge.classList.contains('cat')) cats++;
        });
        
        batchDogCount.textContent = dogs;
        batchCatCount.textContent = cats;
    }

    btnClearBatch.addEventListener('click', () => {
        batchFiles = [];
        batchTableBody.innerHTML = '';
        batchResultsCard.style.display = 'none';
        batchDropzone.style.display = 'flex';
        batchProgressWrapper.style.display = 'none';
        batchFileInput.value = '';
    });

    btnBatchPredict.addEventListener('click', async () => {
        if (batchFiles.length === 0) return;
        
        btnBatchPredict.setAttribute('disabled', 'true');
        btnClearBatch.setAttribute('disabled', 'true');
        batchProgressWrapper.style.display = 'block';
        
        const total = batchFiles.length;
        batchProgressLabel.textContent = `Đang tải các ảnh lên API... (Tổng ${total} tệp)`;
        batchProgressBarFill.style.width = '10%';
        
        // Prepare FormData
        const formData = new FormData();
        batchFiles.forEach(file => {
            formData.append('files', file);
        });
        formData.append('model_name', modelSelector.value);
        formData.append('threshold', DEFAULT_THRESHOLD);
        
        try {
            batchProgressBarFill.style.width = '30%';
            batchProgressLabel.textContent = 'Mô hình đang chạy suy luận (Inference)...';
            
            const response = await fetch(`${API_BASE_URL}/api/predict-batch`, {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                batchProgressBarFill.style.width = '80%';
                batchProgressLabel.textContent = 'Đang nhận kết quả từ API...';
                
                const data = await response.json();
                if (data.success && data.results) {
                    batchProgressBarFill.style.width = '100%';
                    batchProgressLabel.textContent = 'Đã hoàn thành phân loại tất cả ảnh!';
                    
                    // Render results to batch table
                    data.results.forEach((res, index) => {
                        const tr = document.getElementById(`batch-row-${index}`);
                        if (!tr) return;
                        
                        if (res.success) {
                            const badgeClass = res.label.toLowerCase();
                            tr.innerHTML = `
                                <td><img src="${tr.querySelector('img').src}" class="batch-thumbnail"></td>
                                <td style="font-weight: 500; font-family: monospace;">${res.filename}</td>
                                <td>${res.original_width}x${res.original_height} px</td>
                                <td style="text-align: center;">
                                    <span class="batch-class-badge ${badgeClass}">${res.label_vi}</span>
                                </td>
                                <td style="text-align: right; font-weight: 600; font-family: monospace;">${res.confidence}%</td>
                                <td style="text-align: right; color: var(--color-on-surface-variant); font-size: 12px; font-family: monospace;">${res.latency_ms} ms</td>
                            `;
                            
                            // Also add each successful prediction to sidebar history!
                            addToHistory({
                                filename: res.filename,
                                thumbnail: tr.querySelector('img').src,
                                label: res.label,
                                label_vi: res.label_vi,
                                confidence: res.confidence,
                                raw_score: res.raw_score,
                                latency_ms: res.latency_ms,
                                model_used: data.model_used,
                                timestamp: new Date().toLocaleTimeString()
                            });
                        } else {
                            tr.innerHTML = `
                                <td><img src="${tr.querySelector('img').src}" class="batch-thumbnail"></td>
                                <td style="font-weight: 500; font-family: monospace; color: var(--color-error);">${res.filename}</td>
                                <td>Lỗi</td>
                                <td style="text-align: center;"><span class="chip" style="background-color: var(--color-error-container); color: var(--color-on-error-container);">Thất bại</span></td>
                                <td style="text-align: right;">-</td>
                                <td style="text-align: right;">-</td>
                            `;
                        }
                    });
                    
                    updateBatchCounters();
                } else {
                    alert('Lỗi phân tích hàng loạt: ' + data.detail);
                }
            } else {
                alert('Mã phản hồi từ máy chủ không hợp lệ.');
            }
        } catch (error) {
            alert('Lỗi kết nối máy chủ phân tích hàng loạt: ' + error.message);
        } finally {
            btnBatchPredict.removeAttribute('disabled');
            btnClearBatch.removeAttribute('disabled');
            setTimeout(() => {
                batchProgressWrapper.style.display = 'none';
            }, 3000);
        }
    });

    // ==========================================================================
    // 7. Navigation Tabs Handler
    // ==========================================================================
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Run health check on load
    checkApiConnection();
});
