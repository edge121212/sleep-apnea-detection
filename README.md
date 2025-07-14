# Sleep Apnea Detection using ECG Data

這是一個基於心電圖（ECG）數據檢測睡眠呼吸中止症的深度學習項目。該項目使用 Apnea-ECG Database 數據集，結合多並行卷積注意力機制（MPCA）和 Transformer 架構來進行分類。

## 項目特色

- **多並行卷積注意力（MPCA）模塊**：使用不同核大小的並行卷積分支提取多尺度特徵
- **SE注意力機制**：通過 Squeeze-and-Excitation 模塊增強特徵表示
- **Transformer編碼器**：捕捉時間序列中的長期依賴關係
- **Focal Loss**：解決類別不平衡問題
- **多特徵融合**：結合 ECG、R峰振幅（RA）、R-R間期（RRI）和 RRI差分（RRID）

## 數據集

本項目使用 MIT-BIH Apnea-ECG Database：
- 35個有標註的記錄（a01-a20, b01-b05, c01-c10）
- 每分鐘標註為正常（N）或呼吸中止（A）
- 採樣頻率：100 Hz
- 數據長度：約8小時/記錄

## 模型架構

### 1. MPCA Block（多並行卷積注意力）
- 三個並行的卷積分支（核大小：15, 13, 11）
- SE注意力機制
- 簡單路徑進行殘差連接

### 2. Transformer Encoder
- 位置編碼
- 多頭自注意力機制
- 前饋神經網絡

### 3. 分類器
- 全局平均池化
- 全連接層輸出2分類結果

## 數據預處理

1. **小波去噪**：使用 Coiflets 4 小波去除高頻和低頻噪聲
2. **R峰檢測**：使用 Hamilton 演算法檢測 R 峰
3. **特徵提取**：
   - ECG：原始心電信號
   - RA：R峰振幅時間序列
   - RRI：R-R間期時間序列
   - RRID：RRI差分時間序列
4. **時間窗分割**：每個樣本為6分鐘窗口（中心分鐘前後各2.5/3.5分鐘）
5. **重採樣**：統一調整到1024個時間點

## 安裝要求

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 數據預處理

首先需要下載 Apnea-ECG Database 並解壓到項目目錄：

```python
# 運行數據預處理
python data_process.py
```

這會生成以下文件：
- `trainX.npy`, `trainY.npy`：訓練數據
- `valX.npy`, `valY.npy`：驗證數據
- `testX.npy`, `testY.npy`：測試數據

### 2. 模型訓練

```python
# 運行模型訓練
python apnea_model.py
```

## 文件結構

```
apnea/
├── README.md                    # 項目說明
├── requirements.txt             # 依賴套件
├── apnea_model.py              # 主要模型定義和訓練代碼
├── data_process.py             # 數據預處理腳本
├── apnea-ecg-database-1.0.0/   # 原始數據目錄
│   ├── a01.dat, a01.hea, a01.apn
│   ├── ...
│   └── split_data/             # 預處理後的數據
│       ├── trainX.npy
│       ├── trainY.npy
│       ├── valX.npy
│       ├── valY.npy
│       ├── testX.npy
│       └── testY.npy
└── .gitignore                  # Git忽略文件
```

## 模型性能

模型會輸出以下指標：
- Accuracy（準確率）
- Precision（精確率）
- Recall（召回率）
- F1-Score
- AUROC（ROC曲線下面積）

## 技術細節

### 損失函數
- **Focal Loss**：γ=2，有效處理類別不平衡問題

### 優化器
- **Adam**：學習率 1e-3

### 正則化
- Dropout（0.1）
- Layer Normalization

## 貢獻

歡迎提交 Issue 和 Pull Request 來改進這個項目。

## 授權

本項目僅供學術研究使用。數據集的使用請遵循 MIT-BIH Arrhythmia Database 的授權條款。

## 參考文獻

1. Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220; 2000.
2. Penzel T, et al. The Apnea-ECG Database. Computers in Cardiology 2000;27:255-258.
