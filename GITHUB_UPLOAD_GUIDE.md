# GitHub 上傳指南

## 📂 如何將此項目上傳到 GitHub

### 步驟 1: 在 GitHub 上創建新倉庫

1. 登錄你的 [GitHub 帳戶](https://github.com)
2. 點擊右上角的 "+" 號，選擇 "New repository"
3. 填寫倉庫信息：
   - **Repository name**: `sleep-apnea-detection` (或你喜歡的名字)
   - **Description**: `Sleep Apnea Detection using ECG Data with MPCA+Transformer Architecture`
   - 選擇 **Public** 或 **Private**
   - **不要** 勾選 "Initialize this repository with a README"
4. 點擊 "Create repository"

### 步驟 2: 將本地項目推送到 GitHub

在項目根目錄（`C:\python\apnea`）執行以下命令：

```bash
# 設置遠程倉庫 (替換成你的 GitHub 用戶名)
git remote add origin https://github.com/你的用戶名/sleep-apnea-detection.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

### 步驟 3: 驗證上傳

1. 回到你的 GitHub 倉庫頁面
2. 刷新頁面，應該能看到所有文件
3. 確認以下文件都已上傳：
   - ✅ README.md
   - ✅ requirements.txt
   - ✅ .gitignore
   - ✅ config.py
   - ✅ utils.py
   - ✅ apnea_model.py
   - ✅ data_process.py
   - ✅ train.py
   - ✅ QUICKSTART.md

### 🚨 重要注意事項

#### 數據文件會被忽略
由於 `.gitignore` 設置，以下文件不會上傳到 GitHub（這是正確的）：
- `apnea-ecg-database-1.0.0/` 文件夾（數據集太大）
- `*.npy` 文件（預處理後的數據）
- `results/` 文件夾（訓練結果）

#### 數據獲取說明
在 README.md 中已經說明了如何獲取數據：
1. 下載 Apnea-ECG Database
2. 解壓到項目根目錄
3. 運行 `python data_process.py` 進行預處理

### 步驟 4: 添加協作者（可選）

如果你想邀請其他人協作：
1. 在倉庫頁面點擊 "Settings"
2. 左側選擇 "Manage access"
3. 點擊 "Invite a collaborator"

### 步驟 5: 設置 GitHub Pages（可選）

如果想要創建項目網站：
1. 在倉庫設置中找到 "Pages"
2. 選擇 "Deploy from a branch"
3. 選擇 "main" 分支
4. README.md 會自動成為首頁

### 常見問題

**Q: 推送時要求輸入密碼怎麼辦？**
A: GitHub 現在使用 Personal Access Token，在設置中生成 token 替代密碼。

**Q: 數據文件太大無法上傳怎麼辦？**
A: 這是正常的，`.gitignore` 已經設置忽略數據文件。用戶需要自己下載數據集。

**Q: 如何更新項目？**
A: 修改代碼後，使用：
```bash
git add .
git commit -m "更新描述"
git push
```

### 項目結構預覽

上傳後，你的 GitHub 倉庫會顯示：

```
sleep-apnea-detection/
├── 📄 README.md              # 項目主頁
├── 📄 QUICKSTART.md           # 快速開始指南  
├── 📄 requirements.txt        # 依賴列表
├── 📄 .gitignore             # Git 忽略文件
├── 🐍 config.py              # 配置參數
├── 🐍 utils.py               # 工具函數
├── 🐍 apnea_model.py         # 模型定義
├── 🐍 data_process.py        # 數據預處理
└── 🐍 train.py               # 訓練腳本
```

### 下一步

上傳成功後，你可以：
1. 在 README.md 中添加項目徽章
2. 創建 Issues 來追蹤待辦事項
3. 使用 Actions 設置自動化測試
4. 邀請其他研究者協作

祝你的項目在 GitHub 上成功！🎉
