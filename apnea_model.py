import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # 引入 tqdm 用於視覺化進度條

trainX_path = r"C:\python\Apnea\physionet\split_data\trainX.npy"
trainY_path = r"C:\python\Apnea\physionet\split_data\trainY.npy"
valX_path = r"C:\python\Apnea\physionet\split_data\valX.npy"
valY_path = r"C:\python\Apnea\physionet\split_data\valY.npy"
testX_path = r"C:\python\Apnea\physionet\split_data\testX.npy"
testY_path = r"C:\python\Apnea\physionet\split_data\testY.npy"

def load_data(batch_size=64):
    X_train = torch.tensor(np.load(trainX_path), dtype=torch.float32)
    y_train = torch.tensor(np.load(trainY_path), dtype=torch.long)
    X_val = torch.tensor(np.load(valX_path), dtype=torch.float32)
    y_val = torch.tensor(np.load(valY_path), dtype=torch.long)
    X_test = torch.tensor(np.load(testX_path), dtype=torch.float32)
    y_test = torch.tensor(np.load(testY_path), dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)

    def forward(self, x):
        b, c, t = x.size()
        y = self.gap(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y.expand_as(x)

class MPCA_Block(nn.Module):
    def __init__(self):
        super(MPCA_Block, self).__init__()

        # 三個並行的卷積分支
        self.branch1 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 添加 Max Pooling
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 添加 Max Pooling
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=13, padding=6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # 拼接後的點卷積
        self.concat_conv = nn.Conv1d(192, 64, kernel_size=1)

        # SE 模塊
        self.se = SEBlock(192)

        # 簡單路徑
        self.simple_path = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=1),  # Point-wise Conv1
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 第一次降維：1024 → 512
            nn.Conv1d(32, 64, kernel_size=1),  # Point-wise Conv2
            nn.MaxPool1d(kernel_size=2, stride=2),  # 第二次降維：512 → 256
        )

    def forward(self, x):
        x1 = self.branch1(x)  # [batch_size, 64, 256]
        x2 = self.branch2(x)  # [batch_size, 64, 256]
        x3 = self.branch3(x)  # [batch_size, 64, 256]

        x_concat = torch.cat((x1, x2, x3), dim=1)  # [batch_size, 192, 256]
        x_se = self.se(x_concat)
        x_se = self.concat_conv(x_se)  # [batch_size, 64, 256]

        x_simple = self.simple_path(x)  # [batch_size, 64, 256]

        return x_se + x_simple  # [batch_size, 64, 256]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, nhead=4, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        # src shape: [B, T, d_model]
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
    
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.mpca = MPCA_Block()
        self.pos_enc = PositionalEncoding(d_model=64)

        self.transformer_layers = nn.Sequential(
            TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=512),
            TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=512),
            TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=512),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 時間維度上做平均
        self.fc = nn.Linear(64, 2)  # 分成兩類（正常、呼吸中止）

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch_size, 4, 1024] 轉換為 [batch_size, 4, 1024]
        x = self.mpca(x)        # [batch_size, 64, 256]
        x = x.permute(0, 2, 1)  # [batch_size, 256, 64]
        x = self.pos_enc(x)     # [batch_size, 256, 64]
        x = self.transformer_layers(x)  # [batch_size, 256, 64]
        x = x.permute(0, 2, 1)  # [batch_size, 64, 256]
        x = self.global_avg_pool(x)  # [batch_size, 64, 1]
        x = x.squeeze(-1)       # [batch_size, 64]
        x = self.fc(x)          # [batch_size, 2]
        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# 在訓練函數中改用 Focal Loss
def train_model(model, train_loader, val_loader, num_epochs, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = FocalLoss(gamma=2)  # 使用 Focal Loss，gamma=2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 使用 tqdm 包裝 train_loader，顯示進度條
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for X, y in tepoch:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # 更新進度條的描述
                tepoch.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss:.4f}")
        evaluate_model(model, val_loader, device, name="Validation")

def evaluate_model(model, dataloader, device, name="Test"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        # 使用 tqdm 包裝 dataloader，顯示進度條
        with tqdm(dataloader, desc=f"Evaluating {name}", unit="batch") as tepoch:
            for X, y in tepoch:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = float('nan')  # 若無法算 AUROC (如只有一類)

    print(f"{name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUROC: {auc:.4f}")

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()

    model = YourModel()
    train_model(model, train_loader, val_loader, num_epochs=10)  # 測試用 epoch 設為 5

    print("\nFinal Evaluation on Test Set:")
    evaluate_model(model, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
