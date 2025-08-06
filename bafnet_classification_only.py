#!/usr/bin/env python
# coding: utf-8

import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ScaledDotProductAttention(nn.Module):
    def __init__(self, return_attention=False, history_only=False):
        super(ScaledDotProductAttention, self).__init__()
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = None
        self.attention = None

    def forward(self, query, key, value, mask=None):
        """
        簡化的注意力機制 - 處理維度不匹配問題
        Args:
            query: [batch, channels, time]
            key: [batch, channels, time] 
            value: [batch, channels, time]
        """
        # 🔧 解決方案1: 確保所有輸入有相同的時間維度
        batch_size, channels, time_steps = query.shape
        
        # 如果 key 和 value 的時間維度不同，使用自適應池化調整
        if key.shape[2] != time_steps:
            key = F.adaptive_avg_pool1d(key, time_steps)
        if value.shape[2] != time_steps:
            value = F.adaptive_avg_pool1d(value, time_steps)
            
        # 🔧 解決方案2: 使用全局平均池化簡化注意力
        # 將時間維度池化，專注於通道間的注意力
        query_pooled = F.adaptive_avg_pool1d(query, 1).squeeze(-1)  # [batch, channels]
        key_pooled = F.adaptive_avg_pool1d(key, 1).squeeze(-1)      # [batch, channels]
        value_pooled = F.adaptive_avg_pool1d(value, 1).squeeze(-1)  # [batch, channels]
        
        # 計算注意力權重 [batch, channels, channels]
        attention_scores = torch.matmul(query_pooled.unsqueeze(2), key_pooled.unsqueeze(1))
        attention_scores = attention_scores / math.sqrt(channels)
        
        # 軟最大化
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, channels, channels]
        
        # 應用注意力到 value
        attended_value = torch.matmul(attention_weights, value_pooled.unsqueeze(-1)).squeeze(-1)  # [batch, channels]
        
        # 🔧 解決方案3: 將注意力結果廣播回原始時間維度
        # 使用通道注意力的結果來重新加權原始的 value
        attention_mask = F.sigmoid(attended_value).unsqueeze(-1)  # [batch, channels, 1]
        output = value * attention_mask  # [batch, channels, time]
        
        if self.return_attention:
            return output, attention_weights
        return output

class BAFNet_ClassificationOnly(nn.Module):
    """
    純分類版本的 BAFNet - 移除重建部分，專注於分類任務
    """
    def __init__(self, input_shape, weight=1e-3):
        super(BAFNet_ClassificationOnly, self).__init__()
        
        # 修改：適配4個通道 (ECG, RA, RRI, RRID)
        # Initial convolutions - 分別處理四個通道
        self.conv1_ch1 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=True)  # ECG
        self.conv1_ch2 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=True)  # RA
        self.conv1_ch3 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=True)  # RRI
        self.conv1_ch4 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=True)  # RRID
        
        # Second layer convolutions
        self.conv2_ch1 = nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5, bias=True)
        self.conv2_ch2 = nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5, bias=True)
        self.conv2_ch3 = nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5, bias=True)
        self.conv2_ch4 = nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5, bias=True)
        self.pool1 = nn.MaxPool1d(3, padding=1)
        
        # Third layer convolutions
        self.conv3_ch1 = nn.Conv1d(24, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.conv3_ch2 = nn.Conv1d(24, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.conv3_ch3 = nn.Conv1d(24, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.conv3_ch4 = nn.Conv1d(24, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.pool2 = nn.MaxPool1d(5, padding=2)
        
        # 修改：融合層適配4個通道 (24*4 = 96)
        self.conv_fusion = nn.Conv1d(96, 32, kernel_size=11, stride=1, padding=5, bias=True)
        
        # Attention layers
        self.attention = ScaledDotProductAttention()
        
        # 🗑️ 移除：完全移除重建部分 (up1_ch1, up1_ch2, up1_ch3, up1_ch4)
        # 這些佔了大量參數但對分類沒有直接幫助
        
        # ✨ 增強：更強的分類頭部
        self.fc1 = nn.Linear(128, 128, bias=True)
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.dropout = nn.Dropout(0.5)  # 添加 Dropout 來替代重建的正則化效果
        
        # Classification head - 更深的分類器
        self.classifier = nn.Sequential(
            nn.Linear(64, 32, bias=True),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2, bias=True)
        )
        
        # Store weight decay parameter
        self.weight_decay = weight
        
    def forward(self, x):
        # 添加 Min-Max 標準化（與 YourModel 一致）
        x_normalized = torch.zeros_like(x)
        for i in range(4):
            channel_data = x[:, :, i]
            min_val = channel_data.min(dim=1, keepdim=True)[0]
            max_val = channel_data.max(dim=1, keepdim=True)[0]
            range_val = max_val - min_val + 1e-8
            x_normalized[:, :, i] = (channel_data - min_val) / range_val
    
        # 編碼器部分（保持不變）
        x = x_normalized.permute(0, 2, 1)  # [batch_size, 4, 1024]
        
        # 分離四個通道
        ch1 = x[:, 0:1, :]  # ECG   [batch_size, 1, 1024]
        ch2 = x[:, 1:2, :]  # RA    [batch_size, 1, 1024]
        ch3 = x[:, 2:3, :]  # RRI   [batch_size, 1, 1024]
        ch4 = x[:, 3:4, :]  # RRID  [batch_size, 1, 1024]
        
        # Encoder path - 每個通道獨立處理
        x1 = F.relu(self.conv1_ch1(ch1))
        x2 = F.relu(self.conv1_ch2(ch2))
        x3 = F.relu(self.conv1_ch3(ch3))
        x4 = F.relu(self.conv1_ch4(ch4))
        
        x1 = F.relu(self.conv2_ch1(x1))
        x1 = self.pool1(x1)
        x2 = F.relu(self.conv2_ch2(x2))
        x2 = self.pool1(x2)
        x3 = F.relu(self.conv2_ch3(x3))
        x3 = self.pool1(x3)
        x4 = F.relu(self.conv2_ch4(x4))
        x4 = self.pool1(x4)
        
        # 第二層融合
        fsn2 = torch.cat([x1, x2, x3, x4], dim=1)  # [batch_size, 96, time]
        
        x1 = F.relu(self.conv3_ch1(x1))
        x1 = self.pool2(x1)
        x2 = F.relu(self.conv3_ch2(x2))
        x2 = self.pool2(x2)
        x3 = F.relu(self.conv3_ch3(x3))
        x3 = self.pool2(x3)
        x4 = F.relu(self.conv3_ch4(x4))
        x4 = self.pool2(x4)
        
        fsn3 = F.relu(self.conv_fusion(fsn2))
        fsn3 = self.pool2(fsn3)
        
        # Attention mechanism
        fsn3 = self.attention(fsn3, fsn3, fsn3)
        x1 = self.attention(fsn3, x1, x1)
        x2 = self.attention(fsn3, x2, x2)
        x3 = self.attention(fsn3, x3, x3)
        x4 = self.attention(fsn3, x4, x4)
        
        # Concatenate features
        concat = torch.cat([x1, x2, x3, x4], dim=1)  # [batch_size, 128, time]
        
        # ✨ 增強的特徵融合
        # 1. 全局平均池化
        global_avg = torch.mean(concat, dim=2)  # [batch_size, 128]
        
        # 2. 全局最大池化 
        global_max = torch.max(concat, dim=2)[0]  # [batch_size, 128]
        
        # 3. 組合特徵
        combined_features = global_avg + global_max  # [batch_size, 128]
        
        # ✨ 增強的分類路徑
        x = F.relu(self.fc1(combined_features))  # [batch_size, 128]
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # [batch_size, 64]
        x = self.dropout(x)
        outputs = self.classifier(x)  # [batch_size, 2]
        
        # 🔄 只返回分類結果，沒有重建輸出
        return outputs

class ApneaDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_apnea_data():
    """
    載入處理好的睡眠呼吸中止數據
    預期數據格式: [batch_size, 1024, 4]
    """
    try:
        base_dir = r"C:\python\Apnea\physionet\split_data"
        
        # 載入數據
        x_train = np.load(os.path.join(base_dir, "trainX.npy"))
        y_train = np.load(os.path.join(base_dir, "trainY.npy"))
        x_val = np.load(os.path.join(base_dir, "valX.npy"))
        y_val = np.load(os.path.join(base_dir, "valY.npy"))
        x_test = np.load(os.path.join(base_dir, "testX.npy"))
        y_test = np.load(os.path.join(base_dir, "testY.npy"))
        
        print(f"數據載入成功!")
        print(f"訓練集: {x_train.shape}, 標籤: {y_train.shape}")
        print(f"驗證集: {x_val.shape}, 標籤: {y_val.shape}")
        print(f"測試集: {x_test.shape}, 標籤: {y_test.shape}")
        print(f"標籤範圍: {np.unique(y_train)}")
        
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
        
        return x_train, y_train, x_val, y_val, x_test, y_test
        
    except Exception as e:
        print(f"數據載入失敗: {e}")
        print("請確保數據文件存在於指定路徑")
        return None

def train_model_classification_only(model, train_loader, val_loader, optimizer, num_epochs, device, early_stopping_patience=10):
    """
    純分類訓練 - 沒有重建損失
    
    Args:
        early_stopping_patience: 早停耐心值，驗證損失連續多少個epoch不改善就停止
    """
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # 只使用分類損失
    classification_criterion = nn.CrossEntropyLoss()
    
    print(f"開始訓練 {num_epochs} epochs (含早停機制，耐心值={early_stopping_patience})")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', total=len(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # 只有分類輸出
            
            # 只計算分類損失
            cls_loss = classification_criterion(outputs, targets)
            
            # L2 正則化
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            
            # 總損失（沒有重建損失！）
            loss = cls_loss + model.weight_decay * l2_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)  # 只有分類輸出
                
                cls_loss = classification_criterion(outputs, targets)
                
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                
                loss = cls_loss + model.weight_decay * l2_reg
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'BAFNet_ClassificationOnly_best.pth')
            print(f"💾 保存最佳模型 (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            
        # 早停檢查
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 早停觸發！驗證準確率連續 {early_stopping_patience} epochs 沒有改善")
            print(f"最佳驗證準確率: {best_val_acc:.2f}%")
            break
            
    print(f"\n✅ 訓練完成！最佳驗證準確率: {best_val_acc:.2f}%")
    return history

def main():
    print("=== BAFNet 純分類版本訓練 ===")
    
    # 載入數據
    apnea_data = load_apnea_data()
    if apnea_data is None:
        return
        
    x_train, y_train, x_val, y_val, x_test, y_test = apnea_data
    
    # Create datasets and dataloaders
    train_dataset = ApneaDataset(x_train, y_train)
    val_dataset = ApneaDataset(x_val, y_val)
    test_dataset = ApneaDataset(x_test, y_test)
    
    # 🔧 調整 batch size 與 YourModel 一致
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model
    model = BAFNet_ClassificationOnly(input_shape=x_train.shape[1:]).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"訓練樣本數: {len(x_train):,}")
    print(f"每 epoch 批次數: {len(train_loader)}")
    
    # 🎯 Epoch 設定說明
    print("\n📊 Epoch 設定指南:")
    print("• 快速測試: 5-10 epochs")
    print("• 初步訓練: 20-30 epochs") 
    print("• 正式訓練: 50-100 epochs")
    print("• 深度優化: 100+ epochs")
    print(f"• 當前設定: 50 epochs (含早停)")
    
    # Train model
    # 🔧 調整 epochs：睡眠呼吸中止檢測建議 50-100 epochs
    # 初期測試: 20 epochs, 正式訓練: 50-100 epochs
    history = train_model_classification_only(model, train_loader, val_loader, optimizer, 
                                            num_epochs=50, device=device, 
                                            early_stopping_patience=10)
    
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # 只有分類輸出
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Calculate metrics
    f1 = f1_score(all_targets, all_preds, average='binary')
    roc = roc_auc_score(all_targets, all_preds)
    
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc:.4f}')
    
    print("\n=== 與原版 BAFNet 的差異 ===")
    print("✅ 移除了 4 個重建解碼器 (大幅減少參數)")
    print("✅ 移除了重建損失 (專注分類)")
    print("✅ 增強了分類頭部 (更深的網絡)")
    print("✅ 添加了 Dropout (替代重建的正則化)")
    print("✅ 組合了全局平均池化和最大池化")
    print("✅ 調整 batch size 與 YourModel 一致")

if __name__ == "__main__":
    main()
