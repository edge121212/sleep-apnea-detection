# 快速開始指南

## 🚀 快速開始

### 1. 克隆倉庫
```bash
git clone https://github.com/yourusername/sleep-apnea-detection.git
cd sleep-apnea-detection
```

### 2. 安裝依賴
```bash
pip install -r requirements.txt
```

### 3. 準備數據
下載 Apnea-ECG Database 數據集：
- 訪問 [PhysioNet](https://physionet.org/content/apnea-ecg/1.0.0/)
- 下載完整數據集
- 解壓到項目根目錄，確保有 `apnea-ecg-database-1.0.0/` 文件夾

### 4. 數據預處理
```bash
python data_process.py
```

### 5. 訓練模型
```bash
python train.py
```

### 6. 或者直接運行原始代碼
```bash
python apnea_model.py
```

## 📁 項目結構
```
sleep-apnea-detection/
├── README.md              # 項目說明
├── QUICKSTART.md           # 快速開始指南
├── requirements.txt        # 依賴列表
├── config.py              # 配置文件
├── utils.py               # 工具函數
├── data_process.py        # 數據預處理
├── apnea_model.py         # 原始模型代碼
├── train.py               # 改進的訓練腳本
├── .gitignore             # Git忽略文件
└── results/               # 訓練結果（自動創建）
    ├── best_model.pth
    ├── final_model.pth
    ├── training_history.png
    └── training_results.json
```

## 🔧 自定義配置

修改 `config.py` 中的參數來調整模型或訓練設置：

```python
# 例如：調整訓練參數
BATCH_SIZE = 32        # 減少批次大小
LEARNING_RATE = 5e-4   # 降低學習率
NUM_EPOCHS = 20        # 增加訓練輪數
```

## 📊 查看結果

訓練完成後，在 `results/` 目錄中查看：
- 模型權重檔案
- 訓練曲線圖
- 詳細結果 JSON

## 🐛 常見問題

### Q: CUDA 記憶體不足
A: 降低 `config.py` 中的 `BATCH_SIZE`

### Q: 數據檔案找不到
A: 確認數據集路徑正確，檢查 `config.py` 中的 `BASE_DIR`

### Q: 安裝依賴失敗
A: 建議使用虛擬環境：
```bash
python -m venv apnea_env
apnea_env\Scripts\activate  # Windows
pip install -r requirements.txt
```
