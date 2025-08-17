import numpy as np
import os

base_dir = r"C:\Users\lery9\OneDrive\桌面\摳資待嘎酷\三下\專題\胖子\physionet\沒shuffle、record後20%、Z標準化的split_data"

def undersample(X_path, Y_path, out_X, out_Y):
    X = np.load(X_path)
    Y = np.load(Y_path)
    pos_idx = np.where(Y == 1)[0]
    neg_idx = np.where(Y == 0)[0]
    num_pos = len(pos_idx)
    num_neg = len(neg_idx)
    if num_neg > num_pos:
        np.random.seed(42)
        neg_idx_sampled = np.random.choice(neg_idx, num_pos, replace=False)
    else:
        neg_idx_sampled = neg_idx
    final_idx = np.concatenate([pos_idx, neg_idx_sampled])
    np.random.shuffle(final_idx)
    X_under = X[final_idx]
    Y_under = Y[final_idx]
    np.save(out_X, X_under)
    np.save(out_Y, Y_under)
    print(f"{out_X} / {out_Y} 已完成 undersampling，正負樣本各 {num_pos} 筆")

# train
undersample(
    os.path.join(base_dir, "trainX.npy"),
    os.path.join(base_dir, "trainY.npy"),
    os.path.join(base_dir, "trainX_under.npy"),
    os.path.join(base_dir, "trainY_under.npy")
)

# val
undersample(
    os.path.join(base_dir, "valX.npy"),
    os.path.join(base_dir, "valY.npy"),
    os.path.join(base_dir, "valX_under.npy"),
    os.path.join(base_dir, "valY_under.npy")
)