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

class BAFNet(nn.Module):
    def __init__(self, input_shape, weight=1e-3):
        super(BAFNet, self).__init__()
        
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
        
        # 修改：解碼器適配1024時間點和4個通道
        self.up1_ch1 = nn.Sequential(
            nn.Upsample(size=1024, mode='linear', align_corners=True),
            nn.ConvTranspose1d(32, 24, kernel_size=11, stride=1, padding=5, bias=True),
            nn.Upsample(size=1024, mode='linear', align_corners=True),
            nn.ConvTranspose1d(24, 16, kernel_size=11, stride=1, padding=5, bias=True),
            nn.ConvTranspose1d(16, 1, kernel_size=11, stride=1, padding=5, bias=True)
        )
        
        self.up1_ch2 = nn.Sequential(
            nn.Upsample(size=1024, mode='linear', align_corners=True),
            nn.ConvTranspose1d(32, 24, kernel_size=11, stride=1, padding=5, bias=True),
            nn.Upsample(size=1024, mode='linear', align_corners=True),
            nn.ConvTranspose1d(24, 16, kernel_size=11, stride=1, padding=5, bias=True),
            nn.ConvTranspose1d(16, 1, kernel_size=11, stride=1, padding=5, bias=True)
        )
        
        self.up1_ch3 = nn.Sequential(
            nn.Upsample(size=1024, mode='linear', align_corners=True),
            nn.ConvTranspose1d(32, 24, kernel_size=11, stride=1, padding=5, bias=True),
            nn.Upsample(size=1024, mode='linear', align_corners=True),
            nn.ConvTranspose1d(24, 16, kernel_size=11, stride=1, padding=5, bias=True),
            nn.ConvTranspose1d(16, 1, kernel_size=11, stride=1, padding=5, bias=True)
        )
        
        self.up1_ch4 = nn.Sequential(
            nn.Upsample(size=1024, mode='linear', align_corners=True),
            nn.ConvTranspose1d(32, 24, kernel_size=11, stride=1, padding=5, bias=True),
            nn.Upsample(size=1024, mode='linear', align_corners=True),
            nn.ConvTranspose1d(24, 16, kernel_size=11, stride=1, padding=5, bias=True),
            nn.ConvTranspose1d(16, 1, kernel_size=11, stride=1, padding=5, bias=True)
        )
        
        # 修改：Channel-wise fusion module適配更多特徵 (32*4 = 128)
        self.fc1 = nn.Linear(128, 64, bias=True)
        self.fc2 = nn.Linear(64, 128, bias=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 2, bias=True)  # 修改：適配新的特徵維度
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
    
        # 原來的 BAFNet forward 邏輯
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
        
        # Channel-wise fusion
        squeeze = torch.mean(concat, dim=2)  # [batch_size, 128]
        excitation = F.relu(self.fc1(squeeze))  # [batch_size, 64]
        excitation = torch.sigmoid(self.fc2(excitation))  # [batch_size, 128]
        excitation = excitation.unsqueeze(2)  # [batch_size, 128, 1]
        scale = concat * excitation
        
        # Decoder path - 重建所有4個通道
        fcn1 = self.up1_ch1(x1)  # 重建 ECG
        fcn2 = self.up1_ch2(x2)  # 重建 RA
        fcn3 = self.up1_ch3(x3)  # 重建 RRI
        fcn4 = self.up1_ch4(x4)  # 重建 RRID
        
        # Classification
        x = torch.mean(scale, dim=2)  # [batch_size, 128]
        outputs = self.classifier(x)  # [batch_size, 2]
        
        return outputs, fcn1, fcn2, fcn3, fcn4

class ApneaDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        # 🔧 修正：標籤應該使用 LongTensor，與 YourModel 保持一致
        self.y = torch.LongTensor(y)  # 改為 LongTensor
        
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
        print(f"標籤範圍: {np.unique(y_train)}")  # 檢查標籤格式
        
        # 🔧 修正：確保標籤是正確的格式 (0, 1)，不要轉換為 one-hot
        # 因為 CrossEntropyLoss 需要類別索引，不是 one-hot
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
        
        return x_train, y_train, x_val, y_val, x_test, y_test
        
    except Exception as e:
        print(f"數據載入失敗: {e}")
        print("請確保數據文件存在於指定路徑")
        return None

def load_data():
    base_dir = "."
    ir = 3  # interpolate interval
    before = 2
    after = 2
    
    # normalize
    scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    path = "apnea-ecg.pkl"
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
    
    with open(os.path.join(base_dir, path), 'rb') as f:
        apnea_ecg = pickle.load(f)
        
    x, x_train, x_val = [], [], []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_train[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x.append([rri_interp_signal, ampl_interp_signal])
        
    num = [i for i in range(16709)]
    trainlist, vallist, y_train, y_val = train_test_split(num, y_train, test_size=0.3, random_state=42, stratify=y_train)
    
    groups_train_split = []
    groups_val_split = []
    
    for i in trainlist:
        x_train.append(x[i])
        groups_train_split.append(groups_train[i])
    for i in vallist:
        x_val.append(x[i])
        groups_val_split.append(groups_train[i])
        
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))
    y_train = np.array(y_train, dtype="float32")
    x_val = np.array(x_val, dtype="float32").transpose((0, 2, 1))
    y_val = np.array(y_val, dtype="float32")
    
    # Load test data
    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_test[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
        
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")
    
    return x_train, y_train, groups_train_split, x_val, y_val, groups_val_split, x_test, y_test, groups_test

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', total=len(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)  # 🔧 移除冗餘的 .long()
            
            optimizer.zero_grad()
            outputs, fcn1, fcn2, fcn3, fcn4 = model(inputs)  # 修改：接收4個重建輸出
            
            # Calculate losses
            cls_loss = classification_criterion(outputs, targets)  # 🔧 現在 targets 是類別索引
            # 修改：重建損失針對所有4個通道
            recon_loss1 = reconstruction_criterion(fcn1, inputs[:, :, 0:1].permute(0, 2, 1))  # ECG
            recon_loss2 = reconstruction_criterion(fcn2, inputs[:, :, 1:2].permute(0, 2, 1))  # RA
            recon_loss3 = reconstruction_criterion(fcn3, inputs[:, :, 2:3].permute(0, 2, 1))  # RRI
            recon_loss4 = reconstruction_criterion(fcn4, inputs[:, :, 3:4].permute(0, 2, 1))  # RRID
            
            # Add L2 regularization for weights and biases
            l2_reg = torch.tensor(0., device=device)
            for name, param in model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    l2_reg += torch.norm(param)
            
            # 🔧 修正：降低重建損失的權重，避免壓倒分類損失
            total_recon_loss = recon_loss1 + recon_loss2 + recon_loss3 + recon_loss4
            loss = cls_loss + 0.1 * total_recon_loss + model.weight_decay * l2_reg
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()  # 🔧 修正：直接比較，不用 argmax
            
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # 🔧 移除冗餘的 .long()
                outputs, fcn1, fcn2, fcn3, fcn4 = model(inputs)  # 修改：接收4個重建輸出
                
                # Calculate losses
                cls_loss = classification_criterion(outputs, targets)  # 🔧 現在 targets 是類別索引
                # 修改：重建損失針對所有4個通道
                recon_loss1 = reconstruction_criterion(fcn1, inputs[:, :, 0:1].permute(0, 2, 1))  # ECG
                recon_loss2 = reconstruction_criterion(fcn2, inputs[:, :, 1:2].permute(0, 2, 1))  # RA
                recon_loss3 = reconstruction_criterion(fcn3, inputs[:, :, 2:3].permute(0, 2, 1))  # RRI
                recon_loss4 = reconstruction_criterion(fcn4, inputs[:, :, 3:4].permute(0, 2, 1))  # RRID
                
                # Add L2 regularization for weights and biases
                l2_reg = torch.tensor(0., device=device)
                for name, param in model.named_parameters():
                    if 'weight' in name or 'bias' in name:
                        l2_reg += torch.norm(param)
                
                # 🔧 修正：使用相同的損失權重
                total_recon_loss = recon_loss1 + recon_loss2 + recon_loss3 + recon_loss4
                loss = cls_loss + 0.1 * total_recon_loss + model.weight_decay * l2_reg
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()  # 🔧 修正：直接比較，不用 argmax
                
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
            torch.save(model.state_dict(), 'BAFNet_pytorch.pth')
            
    return history

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].plot(history['train_loss'], 'r-', label='Train')
    axes[0].plot(history['val_loss'], 'b-', label='Validation')
    axes[0].set_title('Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], 'r-', label='Train')
    axes[1].plot(history['val_acc'], 'b-', label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    
    fig.tight_layout()
    plt.show()

def main():
    print("=== BAFNet 訓練 (適配睡眠呼吸中止數據) ===")
    
    # 嘗試載入新的數據格式
    apnea_data = load_apnea_data()
    if apnea_data is not None:
        x_train, y_train, x_val, y_val, x_test, y_test = apnea_data
        print("使用睡眠呼吸中止數據集")
    else:
        print("回退到原始數據載入方式")
        # Load original data
        x_train, y_train, groups_train, x_val, y_val, groups_val, x_test, y_test, groups_test = load_data()
        
        # 🔧 修正：確保標籤是類別索引格式，不要轉換為 one-hot
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
    
    # Create datasets and dataloaders
    train_dataset = ApneaDataset(x_train, y_train)
    val_dataset = ApneaDataset(x_val, y_val)
    test_dataset = ApneaDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    # Initialize model
    model = BAFNet(input_shape=x_train.shape[1:]).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train model
    # 🔧 調整 epochs：複雜的重建+分類模型需要更多訓練時間
    # 建議: 30-50 epochs (重建任務增加訓練複雜度)
    history = train_model(model, train_loader, val_loader, None, optimizer, num_epochs=30, device=device)
    
    # Plot training history
    #plot_history(history)
    
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # 🔧 移除冗餘的 .long()
            outputs, _, _, _, _ = model(inputs)  # 修改：忽略4個重建輸出，只關注分類結果
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()  # 🔧 修正：直接比較，不用 argmax
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())  # 🔧 修正：不用 argmax
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Calculate metrics
    f1 = f1_score(all_targets, all_preds, average='binary')
    roc = roc_auc_score(all_targets, all_preds)
    
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc:.4f}')

if __name__ == "__main__":
    main()