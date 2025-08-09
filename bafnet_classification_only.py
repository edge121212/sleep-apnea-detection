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
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    專門用於改善 F1 score 和 accuracy 之間的差距
    """
    def __init__(self, alpha=1, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

class HybridLoss(nn.Module):
    """
    Hybrid Loss Function 結合多種損失函數
    - Focal Loss: 處理類別不平衡
    - Cross Entropy: 標準分類損失  
    - Dice Loss: 改善 F1 score
    - Label Smoothing: 防止過度自信
    """
    def __init__(self, alpha=1, gamma=2, weight=None, 
                 focal_weight=0.5, ce_weight=0.3, dice_weight=0.2, 
                 label_smoothing=0.1):
        super(HybridLoss, self).__init__()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight  
        self.dice_weight = dice_weight
        self.label_smoothing = label_smoothing
        
        # 各種損失函數
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, weight=weight)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        
        print(f"🔄 Hybrid Loss 權重配置:")
        print(f"  • Focal Loss: {focal_weight}")
        print(f"  • Cross Entropy: {ce_weight}") 
        print(f"  • Dice Loss: {dice_weight}")
        print(f"  • Label Smoothing: {label_smoothing}")
    
    def dice_loss(self, preds, targets):
        """
        Dice Loss 專門改善 F1 score
        """
        # 獲取預測概率
        probs = F.softmax(preds, dim=1)
        
        # 將目標轉換為 one-hot
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        
        # 計算每個類別的 Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice_coeff = (2.0 * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1.0 - dice_coeff.mean()
        
        return dice_loss
    
    def forward(self, preds, labels):
        # 計算各種損失
        focal = self.focal_loss(preds, labels)
        ce = self.ce_loss(preds, labels) 
        dice = self.dice_loss(preds, labels)
        
        # 加權組合
        total_loss = (self.focal_weight * focal + 
                     self.ce_weight * ce + 
                     self.dice_weight * dice)
        
        return total_loss

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
    🛡️ 添加多層次正則化防止過擬合
    """
    def __init__(self, input_shape, weight=1e-3, dropout_rate=0.3):
        super(BAFNet_ClassificationOnly, self).__init__()
        
        # 修改：適配4個通道 (ECG, RA, RRI, RRID)
        # Initial convolutions - 分別處理四個通道
        self.conv1_ch1 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=True)  # ECG
        self.conv1_ch2 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=True)  # RA
        self.conv1_ch3 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=True)  # RRI
        self.conv1_ch4 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=True)  # RRID
        
        # 🛡️ 添加 BatchNorm 和 Dropout 到卷積層
        self.bn1_ch1 = nn.BatchNorm1d(16)
        self.bn1_ch2 = nn.BatchNorm1d(16)
        self.bn1_ch3 = nn.BatchNorm1d(16)
        self.bn1_ch4 = nn.BatchNorm1d(16)
        self.dropout_conv1 = nn.Dropout1d(dropout_rate * 0.5)  # 卷積層較低 dropout
        
        # Second layer convolutions
        self.conv2_ch1 = nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5, bias=True)
        self.conv2_ch2 = nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5, bias=True)
        self.conv2_ch3 = nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5, bias=True)
        self.conv2_ch4 = nn.Conv1d(16, 24, kernel_size=11, stride=2, padding=5, bias=True)
        self.pool1 = nn.MaxPool1d(3, padding=1)
        
        # 🛡️ 第二層 BatchNorm
        self.bn2_ch1 = nn.BatchNorm1d(24)
        self.bn2_ch2 = nn.BatchNorm1d(24)
        self.bn2_ch3 = nn.BatchNorm1d(24)
        self.bn2_ch4 = nn.BatchNorm1d(24)
        self.dropout_conv2 = nn.Dropout1d(dropout_rate * 0.6)
        
        # Third layer convolutions
        self.conv3_ch1 = nn.Conv1d(24, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.conv3_ch2 = nn.Conv1d(24, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.conv3_ch3 = nn.Conv1d(24, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.conv3_ch4 = nn.Conv1d(24, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.pool2 = nn.MaxPool1d(5, padding=2)
        
        # 🛡️ 第三層 BatchNorm
        self.bn3_ch1 = nn.BatchNorm1d(32)
        self.bn3_ch2 = nn.BatchNorm1d(32)
        self.bn3_ch3 = nn.BatchNorm1d(32)
        self.bn3_ch4 = nn.BatchNorm1d(32)
        self.dropout_conv3 = nn.Dropout1d(dropout_rate * 0.7)
        
        # 修改：融合層適配4個通道 (24*4 = 96)
        self.conv_fusion = nn.Conv1d(96, 32, kernel_size=11, stride=1, padding=5, bias=True)
        self.bn_fusion = nn.BatchNorm1d(32)
        
        # Attention layers
        self.attention = ScaledDotProductAttention()
        
        # 🗑️ 移除：完全移除重建部分 (up1_ch1, up1_ch2, up1_ch3, up1_ch4)
        # 這些佔了大量參數但對分類沒有直接幫助
        
        # ✨ 增強：更強的分類頭部 + 多層次正則化
        self.fc1 = nn.Linear(128, 128, bias=True)
        self.bn_fc1 = nn.BatchNorm1d(128)  # 🛡️ 全連接層也加 BatchNorm
        self.dropout1 = nn.Dropout(dropout_rate * 1.2)  # 更高的 dropout
        
        self.fc2 = nn.Linear(128, 64, bias=True)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Classification head - 更深的分類器
        self.classifier = nn.Sequential(
            nn.Linear(64, 32, bias=True),
            nn.BatchNorm1d(32),  # 🛡️ 分類器也加 BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(32, 2, bias=True)
        )
        
        # Store weight decay parameter
        self.weight_decay = weight
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        # 添加 Min-Max 標準化（與 YourModel 一致）
        x_normalized = torch.zeros_like(x)
        for i in range(4):
            channel_data = x[:, :, i]
            min_val = channel_data.min(dim=1, keepdim=True)[0]
            max_val = channel_data.max(dim=1, keepdim=True)[0]
            range_val = max_val - min_val + 1e-8
            x_normalized[:, :, i] = (channel_data - min_val) / range_val
    
        # 編碼器部分（加入正則化）
        x = x_normalized.permute(0, 2, 1)  # [batch_size, 4, 1024]
        
        # 分離四個通道
        ch1 = x[:, 0:1, :]  # ECG   [batch_size, 1, 1024]
        ch2 = x[:, 1:2, :]  # RA    [batch_size, 1, 1024]
        ch3 = x[:, 2:3, :]  # RRI   [batch_size, 1, 1024]
        ch4 = x[:, 3:4, :]  # RRID  [batch_size, 1, 1024]
        
        # 🛡️ 第一層：卷積 + BatchNorm + ReLU + Dropout
        x1 = F.relu(self.bn1_ch1(self.conv1_ch1(ch1)))
        x2 = F.relu(self.bn1_ch2(self.conv1_ch2(ch2)))
        x3 = F.relu(self.bn1_ch3(self.conv1_ch3(ch3)))
        x4 = F.relu(self.bn1_ch4(self.conv1_ch4(ch4)))
        
        # 應用第一層 dropout
        x1 = self.dropout_conv1(x1)
        x2 = self.dropout_conv1(x2)
        x3 = self.dropout_conv1(x3)
        x4 = self.dropout_conv1(x4)
        
        # 🛡️ 第二層：卷積 + BatchNorm + ReLU + Pool + Dropout
        x1 = F.relu(self.bn2_ch1(self.conv2_ch1(x1)))
        x1 = self.pool1(x1)
        x2 = F.relu(self.bn2_ch2(self.conv2_ch2(x2)))
        x2 = self.pool1(x2)
        x3 = F.relu(self.bn2_ch3(self.conv2_ch3(x3)))
        x3 = self.pool1(x3)
        x4 = F.relu(self.bn2_ch4(self.conv2_ch4(x4)))
        x4 = self.pool1(x4)
        
        # 應用第二層 dropout
        x1 = self.dropout_conv2(x1)
        x2 = self.dropout_conv2(x2)
        x3 = self.dropout_conv2(x3)
        x4 = self.dropout_conv2(x4)
        
        # 第二層融合
        fsn2 = torch.cat([x1, x2, x3, x4], dim=1)  # [batch_size, 96, time]
        
        # 🛡️ 第三層：卷積 + BatchNorm + ReLU + Pool + Dropout
        x1 = F.relu(self.bn3_ch1(self.conv3_ch1(x1)))
        x1 = self.pool2(x1)
        x2 = F.relu(self.bn3_ch2(self.conv3_ch2(x2)))
        x2 = self.pool2(x2)
        x3 = F.relu(self.bn3_ch3(self.conv3_ch3(x3)))
        x3 = self.pool2(x3)
        x4 = F.relu(self.bn3_ch4(self.conv3_ch4(x4)))
        x4 = self.pool2(x4)
        
        # 應用第三層 dropout
        x1 = self.dropout_conv3(x1)
        x2 = self.dropout_conv3(x2)
        x3 = self.dropout_conv3(x3)
        x4 = self.dropout_conv3(x4)
        
        # 🛡️ 融合層 + BatchNorm
        fsn3 = F.relu(self.bn_fusion(self.conv_fusion(fsn2)))
        fsn3 = self.pool2(fsn3)
        
        # Attention mechanism
        fsn3 = self.attention(fsn3, fsn3, fsn3)
        x1 = self.attention(fsn3, x1, x1)
        x2 = self.attention(fsn3, x2, x2)
        x3 = self.attention(fsn3, x3, x3)
        x4 = self.attention(fsn3, x4, x4)
        
        # Concatenate features
        concat = torch.cat([x1, x2, x3, x4], dim=1)  # [batch_size, 128, time]
        
        # ✨ 增強的特徵融合（加入噪聲增強）
        # 1. 全局平均池化
        global_avg = torch.mean(concat, dim=2)  # [batch_size, 128]
        
        # 2. 全局最大池化 
        global_max = torch.max(concat, dim=2)[0]  # [batch_size, 128]
        
        # 3. 🛡️ 特徵組合 + 隨機噪聲（訓練時）
        combined_features = global_avg + global_max  # [batch_size, 128]
        
        # 🛡️ 訓練時添加高斯噪聲防止過擬合
        if self.training:
            noise = torch.randn_like(combined_features) * 0.01
            combined_features = combined_features + noise
        
        # ✨ 增強的分類路徑 + 多層次正則化
        x = F.relu(self.bn_fc1(self.fc1(combined_features)))  # [batch_size, 128]
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))  # [batch_size, 64]
        x = self.dropout2(x)
        
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
    純分類訓練 - 沒有重建損失 + 全面正則化策略 + Focal Loss 改善不平衡
    
    Args:
        early_stopping_patience: 早停耐心值，驗證損失連續多少個epoch不改善就停止
    """
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_recall': [], 'val_recall': []}
    
    # 🎯 使用 Focal Loss 改善類別不平衡問題
    # 計算類別權重
    all_targets = []
    for _, targets in train_loader:
        all_targets.extend(targets.numpy())
    all_targets = np.array(all_targets)
    
    class_counts = np.bincount(all_targets)
    total_samples = len(all_targets)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"🎯 類別分布: {class_counts}, 權重: {class_weights.cpu().numpy()}")
    
    # 🔄 使用 Hybrid Loss 組合多種損失函數
    classification_criterion = HybridLoss(
        alpha=1, gamma=2, weight=class_weights,
        focal_weight=0.4,      # Focal Loss 處理不平衡
        ce_weight=0.3,         # Cross Entropy 標準分類  
        dice_weight=0.3,       # Dice Loss 改善 F1
        label_smoothing=0.1    # Label Smoothing 防止過度自信
    )
    print("� 使用 Hybrid Loss 組合 (Focal + CE + Dice + Label Smoothing)")
    
    # 🛡️ 學習率調度器：當驗證損失停止改善時降低學習率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # 🛡️ 混合精度訓練（如果有 GPU）
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print(f"開始訓練 {num_epochs} epochs (含早停機制，耐心值={early_stopping_patience})")
    print(f"🛡️ 正則化策略：BatchNorm + 分層Dropout + 學習率調度 + 高斯噪聲")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        train_targets_list = []
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', total=len(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 🛡️ 混合精度訓練（更高效 + 隱式正則化）
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)  # 只有分類輸出
                    cls_loss = classification_criterion(outputs, targets)
                    
                    # 🛡️ 增強的L2正則化：分層權重衰減
                    l2_reg = torch.tensor(0., device=device)
                    for name, param in model.named_parameters():
                        if 'conv' in name:
                            l2_reg += torch.norm(param) * 0.5  # 卷積層較輕的正則化
                        elif 'fc' in name or 'classifier' in name:
                            l2_reg += torch.norm(param) * 1.0  # 全連接層較重的正則化
                    
                    # 總損失（沒有重建損失！）
                    loss = cls_loss + model.weight_decay * l2_reg
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # CPU 或無混合精度
                outputs = model(inputs)
                cls_loss = classification_criterion(outputs, targets)
                
                # 🛡️ 增強的L2正則化：分層權重衰減
                l2_reg = torch.tensor(0., device=device)
                for name, param in model.named_parameters():
                    if 'conv' in name:
                        l2_reg += torch.norm(param) * 0.5
                    elif 'fc' in name or 'classifier' in name:
                        l2_reg += torch.norm(param) * 1.0
                
                loss = cls_loss + model.weight_decay * l2_reg
                loss.backward()
                
                # 🛡️ 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # 🎯 收集預測結果用於計算 recall
            train_predictions.extend(predicted.cpu().numpy())
            train_targets_list.extend(targets.cpu().numpy())
            
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_recall = recall_score(train_targets_list, train_predictions, average='binary', zero_division=0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets_list = []
        
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
                
                # 🎯 收集預測結果用於計算 recall
                val_predictions.extend(predicted.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_recall = recall_score(val_targets_list, val_predictions, average='binary', zero_division=0)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        # 🛡️ 學習率調度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Recall: {train_recall:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Recall: {val_recall:.4f}')
        print(f'Learning Rate: {current_lr:.2e}')
        
        # Save best model (基於驗證準確率)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'dropout_rate': model.dropout_rate
            }, 'BAFNet_ClassificationOnly_best.pth')
            print(f"💾 保存最佳模型 (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            
        # 🛡️ 增強的早停：考慮學習率下降
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 早停觸發！驗證準確率連續 {early_stopping_patience} epochs 沒有改善")
            print(f"最佳驗證準確率: {best_val_acc:.2f}%")
            break
            
        # 🛡️ 如果學習率太小，也提前停止
        if current_lr < 1e-6:
            print(f"\n🛑 學習率過小 ({current_lr:.2e})，提前停止訓練")
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
    
    # Initialize model with 可調的dropout率
    # 🛡️ 根據數據集大小調整 dropout：較小數據集需要更強正則化
    dataset_size = len(x_train)
    if dataset_size < 5000:
        dropout_rate = 0.5  # 小數據集：較強正則化
    elif dataset_size < 15000:
        dropout_rate = 0.3  # 中數據集：中等正則化  
    else:
        dropout_rate = 0.2  # 大數據集：較輕正則化
        
    model = BAFNet_ClassificationOnly(input_shape=x_train.shape[1:], 
                                     weight=1e-4,  # 🛡️ 降低權重衰減
                                     dropout_rate=dropout_rate).to(device)
    
    # 🛡️ 優化器：添加權重衰減
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"訓練樣本數: {len(x_train):,}")
    print(f"每 epoch 批次數: {len(train_loader)}")
    print(f"🛡️ Dropout 率: {dropout_rate} (基於數據集大小 {dataset_size:,})")
    
    # 🎯 Epoch 設定說明
    print("\n📊 Epoch 設定指南:")
    print("• 快速測試: 5-10 epochs")
    print("• 初步訓練: 20-30 epochs") 
    print("• 正式訓練: 50-100 epochs")
    print("• 深度優化: 100+ epochs")
    print(f"• 當前設定: 50 epochs (含早停)")
    print("🛡️ 正則化策略：BatchNorm + 分層Dropout + 學習率調度 + 梯度裁剪 + 高斯噪聲")
    
    # Train model
    # 🔧 調整 epochs：睡眠呼吸中止檢測建議 50-100 epochs
    # 初期測試: 20 epochs, 正式訓練: 50-100 epochs
    history = train_model_classification_only(model, train_loader, val_loader, optimizer, 
                                            num_epochs=42, device=device, 
                                            early_stopping_patience=10)
    
    # 獲取最佳驗證準確率用於過擬合檢測
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
    
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
    recall = recall_score(all_targets, all_preds, average='binary')
    
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc:.4f}')
    print(f'Recall: {recall:.4f}')
    
    # 🎯 分析 Accuracy vs F1 差距
    acc_f1_gap = test_acc/100 - f1
    print(f'\n📊 Hybrid Loss 性能分析:')
    print(f'Accuracy: {test_acc:.2f}%')
    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Accuracy-F1 差距: {acc_f1_gap:.4f}')
    
    if acc_f1_gap > 0.1:
        print("⚠️  較大的 Accuracy-F1 差距，建議:")
        print("  • 增加 Dice Loss 權重 (當前: 0.3 → 0.4-0.5)")
        print("  • 調整 Focal Loss gamma 參數 (當前: 2 → 3-4)")
        print("  • 減少 Label Smoothing (當前: 0.1 → 0.05)")
    elif acc_f1_gap > 0.05:
        print("⚠️  中等的 Accuracy-F1 差距，建議微調:")
        print("  • 增加 Dice Loss 權重 (當前: 0.3 → 0.35)")
        print("  • 調整損失函數權重比例")
    else:
        print("✅ Accuracy 和 F1 差距較小，Hybrid Loss 效果良好")
        
    print(f'\n🔄 當前 Hybrid Loss 配置效果:')
    print(f'  • Focal Loss (40%): 處理類別不平衡')
    print(f'  • Cross Entropy (30%): 標準分類損失')
    print(f'  • Dice Loss (30%): 改善 F1 score')
    print(f'  • Label Smoothing (10%): 防止過度自信')
    
    print("\n=== 與原版 BAFNet 的差異 ===")
    print("✅ 移除了 4 個重建解碼器 (大幅減少參數)")
    print("✅ 移除了重建損失 (專注分類)")
    print("✅ 增強了分類頭部 (更深的網絡)")
    print("🛡️ 全面正則化策略:")
    print("  • BatchNorm1d (所有卷積層和全連接層)")
    print("  • 分層 Dropout (漸進式增強)")
    print("  • Dropout1d (卷積層專用)")
    print("  • 高斯噪聲 (訓練時特徵增強)")
    print("  • 學習率調度 (ReduceLROnPlateau)")
    print("  • 梯度裁剪 (防止梯度爆炸)")
    print("  • 分層權重衰減 (卷積層vs全連接層)")
    print("  • 混合精度訓練 (GPU可用時)")
    print("  • AdamW優化器 (內建權重衰減)")
    print("🔄 Hybrid Loss Function:")
    print("  • Focal Loss (處理類別不平衡)")
    print("  • Cross Entropy (標準分類)")
    print("  • Dice Loss (改善 F1 score)")
    print("  • Label Smoothing (防止過度自信)")
    print("✅ 調整 batch size 與 YourModel 一致")
    print("🎯 智能 dropout 率選擇 (基於數據集大小)")
    
    print(f"\n🏆 最終測試性能 (Hybrid Loss):")
    print(f"  • 準確率: {test_acc:.2f}%")
    print(f"  • F1分數: {f1:.4f}")
    print(f"  • ROC AUC: {roc:.4f}")
    print(f"  • Recall: {recall:.4f}")
    print(f"  • 模型參數: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  • Dropout率: {model.dropout_rate}")
    print(f"  • 損失函數: Hybrid (Focal+CE+Dice+LS)")
    
    # 🛡️ 過擬合檢測提示
    if test_acc < best_val_acc * 0.95:  # 測試準確率比最好的驗證準確率低 5% 以上
        print("\n⚠️  可能存在輕微過擬合，建議:")
        print("  • 增加 dropout_rate")
        print("  • 增加數據增強")
        print("  • 調整 Hybrid Loss 權重")
        print("  • 增加 Label Smoothing")
    else:
        print("\n✅ 模型泛化良好，無明顯過擬合")
        
    print(f"\n💡 Hybrid Loss 調優建議:")
    if f1 < 0.8:
        print("  🎯 F1 偏低，建議增加 Dice Loss 權重")
        print("  🔧 可嘗試: dice_weight=0.4, focal_weight=0.3, ce_weight=0.3")
    if recall < 0.8:
        print("  🎯 Recall 偏低，建議增加 Focal Loss gamma")
        print("  🔧 可嘗試: gamma=3 或 4")
    if acc_f1_gap > 0.1:
        print("  🎯 差距太大，建議平衡各損失權重") 
        print("  🔧 可嘗試: focal=0.3, ce=0.2, dice=0.5, label_smoothing=0.05")

if __name__ == "__main__":
    main()
