"""
my_apnea_processing.py

Apnea-ECG data preprocessing:
 - Local alignment of R-peak indices (RA, RRI, RRID) in each 6-min segment
 - Record-level splitting into train, val, test sets
 - Saves output as .npy
"""

import os
import random
import numpy as np
import wfdb
import pywt
from scipy import signal

# ============================
# PART A: DATA READING
# ============================

def load_apnea_ecg_record(record_path):
    """
    Read a single record (ECG) from Apnea-ECG Database using WFDB.
    Returns:
      ecg (np.ndarray): shape (num_samples,)
      fs (int or float): sampling frequency
    """
    signals, fields = wfdb.rdsamp(record_path)
    ecg = signals[:, 0]  # single-lead ECG
    fs = fields["fs"]    # typically 100 Hz
    return ecg, fs


def load_apnea_annotations(ann_path):
    """
    Read minute-by-minute apnea annotations (apn).
    Returns:
      labels (np.ndarray of 0/1), length = # minutes
    """
    ann = wfdb.rdann(ann_path, "apn")
    labels = np.array([1 if s == "A" else 0 for s in ann.symbol], dtype=int)
    return labels

# ============================
# PART B: WAVELET DENOISING
# ============================

def wavelet_denoise(ecg, wavelet="coif4", level=6):
    """
    Remove high freq (~25-50Hz) and very low freq (<1Hz) by zeroing cD1 & cA6.
    """
    coeffs = pywt.wavedec(ecg, wavelet, level=level)
    # cA6 => coeffs[0], cD1 => coeffs[-1]
    coeffs[0]  = np.zeros_like(coeffs[0])
    coeffs[-1] = np.zeros_like(coeffs[-1])
    denoised = pywt.waverec(coeffs, wavelet)
    # match original length
    return denoised[: len(ecg)]

# ============================
# PART C: NORMALIZATION
# ============================

def zscore_normalize(ecg):
    mu = np.mean(ecg)
    sigma = np.std(ecg)
    if sigma < 1e-12:
        return ecg - mu
    return (ecg - mu) / sigma

def minmax_normalize(signal):
    """
    Z-score 標準化：將信號轉換為 (signal - mean) / std
    對每個 segment 獨立進行標準化
    """
    mean = np.mean(signal)
    std = np.std(signal)
    # 避免除零錯誤
    if std < 1e-12:
        return signal - mean  # 如果信號是常數，返回零信號
    return (signal - mean) / std

# ============================
# PART D: R-PEAK DETECTION, AUX
# ============================

def detect_rpeaks(ecg, fs):
    """
    使用 biosppy Hamilton 演算法檢測 R 峰
    """
    print(f"R峰檢測 - 信號長度: {len(ecg)}, 採樣率: {fs} Hz")
    
    from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
    print("使用 biosppy Hamilton 演算法...")
    
    # Hamilton 演算法檢測 R 峰
    rpeaks = hamilton_segmenter(ecg, sampling_rate=fs)[0]
    print(f"   初始檢測到 {len(rpeaks)} 個 R 峰")
    
    # 修正 R 峰位置
    rpeaks_corrected = correct_rpeaks(signal=ecg, 
                                    rpeaks=rpeaks, 
                                    sampling_rate=fs, 
                                    tol=0.05)[0]
    print(f"   修正後有 {len(rpeaks_corrected)} 個 R 峰")
    
    # 計算平均心率
    if len(rpeaks_corrected) > 1:
        avg_rr = np.mean(np.diff(rpeaks_corrected)) / fs
        avg_hr = 60 / avg_rr
        print(f"   平均心率: {avg_hr:.1f} bpm")
    
    return rpeaks_corrected

def construct_ra_rri_rrid(ecg, rpeaks, fs):

    rpeak_times = rpeaks / fs
    ra_values   = ecg[rpeaks]         # amplitude at each R-peak
    rr_intervals= np.diff(rpeak_times)
    rrid        = np.diff(rr_intervals)
    return ra_values, rr_intervals, rrid

# ============================
# PART E: SEGMENTING INTO 6-MIN WINDOWS
# ============================

def segment_ecg_with_context(ecg, fs, labels=None):
    """
    Return a list of (ecg_segment, label, start_sample, end_sample).
    Each segment = 6 min => from [m_idx-2.5..m_idx+3.5).
    If labels is None, generate segments for every minute.
    """
    segments = []
    n_samples = len(ecg)
    minute_len = 60 * fs  # Number of samples in one minute

    if labels is not None:
        # Generate labeled segments
        for m_idx, lbl in enumerate(labels):
            start_sample = int((m_idx - 2.5) * minute_len)
            end_sample = int((m_idx + 3.5) * minute_len)
            if start_sample < 0 or end_sample > n_samples:
                continue
            seg = ecg[start_sample:end_sample]
            segments.append((seg, lbl, start_sample, end_sample))
    else:
        # Generate unlabeled segments for every minute
        num_minutes = n_samples // minute_len
        for m_idx in range(num_minutes):
            start_sample = int((m_idx - 2.5) * minute_len)
            end_sample = int((m_idx + 3.5) * minute_len)
            if start_sample < 0 or end_sample > n_samples:
                continue
            seg = ecg[start_sample:end_sample]
            segments.append((seg, None, start_sample, end_sample))

    return segments

# ============================
# PART F: INTERPOLATION, AUX timeseries
# ============================

def interp_to_length(signal_in, out_len=1024):
    """ Resample 1D array to out_len using FFT-based method """
    return signal.resample(signal_in, out_len)

def build_ra_timeseries(local_ecg, local_rpeaks, length):
    """
    local_ecg: shape (length,) - the 6-min ECG segment
    local_rpeaks: list of R-peak indices in [0, length)
    returns array shape (length,) with amplitude at local_rpeaks
    """
    ra = np.zeros(length)
    for rp in local_rpeaks:
        if 0 <= rp < length:
            ra[rp] = local_ecg[rp]
    return ra

def build_rri_timeseries(local_rpeaks, fs, length):
    """
    For each consecutive R-peaks [i-1, i], fill from [start..end]
    with the same RR interval in seconds. 
    local_rpeaks: sorted list of indices in [0, length)
    """
    rri_series = np.zeros(length)
    for i in range(1, len(local_rpeaks)):
        start = local_rpeaks[i-1]
        end   = local_rpeaks[i]
        rr_val = (end - start) / fs
        if end <= length:
            rri_series[start:end] = rr_val
    return rri_series

def build_rrid_timeseries(local_rpeaks, fs, length):
    """
    first difference of RRI. We'll store it at the R-peaks themselves 
    or fill from [rpeaks[i]..rpeaks[i+1]] with the same RRID, your choice.
    This is a naive approach.
    """
    intervals = []
    for i in range(1, len(local_rpeaks)):
        delta = (local_rpeaks[i] - local_rpeaks[i-1]) / fs
        intervals.append(delta)
    diffs = np.diff(intervals)

    rrid_series = np.zeros(length)
    # place each diff at the peak index i
    for i in range(1, len(intervals)-1):
        idx = local_rpeaks[i]
        if idx < length:
            rrid_series[idx] = diffs[i-1]
    return rrid_series

def preprocess_record(record_id, base_dir, out_len=1024):
    """Modified to return record_id with each segment"""
    print(f"\n處理記錄: {record_id}")
    
    rec_path = os.path.join(base_dir, record_id)
    ecg_raw, fs = load_apnea_ecg_record(rec_path)
    print(f"   原始 ECG 長度: {len(ecg_raw)} 樣本 ({len(ecg_raw)/fs/60:.1f} 分鐘)")

    # Only do wavelet denoising, remove zscore normalization
    ecg_denoised = wavelet_denoise(ecg_raw, "coif4", 6)

    ann_path = os.path.join(base_dir, record_id)
    try:
        labels = load_apnea_annotations(ann_path)
        print(f"   載入標註: {len(labels)} 分鐘, {sum(labels)} 個呼吸中止事件")
    except:
        labels = None
        print("   警告: 無標註文件")

    # Use denoised ECG (not normalized) for r-peak detection
    rpeaks = detect_rpeaks(ecg_denoised, fs)

    data_segments = []
    segments = segment_ecg_with_context(ecg_denoised, fs, labels)
    print(f"   生成 {len(segments)} 個 6 分鐘段")
    print(f"   📊 將對每個 segment 進行 Min-Max 標準化 [0,1]")
    
    for seg_ecg, seg_lbl, start_sample, end_sample in segments:
        seg_len = len(seg_ecg)
        # Resample the local ECG to out_len
        ecg_resamp = interp_to_length(seg_ecg, out_len)

        # 1) Build local r-peaks in [0, seg_len)
        local_rpeaks = [rp - start_sample for rp in rpeaks
                        if (rp >= start_sample) and (rp < end_sample)]

        # === 新增：過濾異常 RR interval 的 R-peak ===
        if len(local_rpeaks) > 1:
            rr_intervals = np.diff(local_rpeaks) / fs  # 單位：秒
            hr = 60 / rr_intervals
            # 合理 HR 範圍 40~180
            valid_idx = np.where((hr >= 40) & (hr <= 180))[0]
            # 只保留合理的 R-peak（注意 diff 會少一個，需補回第一個 R-peak）
            filtered_rpeaks = [local_rpeaks[0]] + [local_rpeaks[i+1] for i in valid_idx]
        else:
            filtered_rpeaks = local_rpeaks

        # 2) RA, RRI, RRID as local time series of length seg_len
        RA_ts  = build_ra_timeseries(seg_ecg, filtered_rpeaks, seg_len)
        RRI_ts = build_rri_timeseries(filtered_rpeaks, fs, seg_len)
        RRID_ts= build_rrid_timeseries(filtered_rpeaks, fs, seg_len)

        # 3) Resample each to out_len
        RA_resamp   = interp_to_length(RA_ts,   out_len)
        RRI_resamp  = interp_to_length(RRI_ts,  out_len)
        RRID_resamp = interp_to_length(RRID_ts, out_len)

        # 🔧 新增：對每個通道進行 Min-Max 標準化 (segment-wise)
        ecg_normalized = minmax_normalize(ecg_resamp)
        RA_normalized = minmax_normalize(RA_resamp)
        RRI_normalized = minmax_normalize(RRI_resamp)
        RRID_normalized = minmax_normalize(RRID_resamp)

        # stack => shape (out_len, 4) 使用標準化後的數據
        seg_4ch = np.vstack([ecg_normalized, RA_normalized, RRI_normalized, RRID_normalized]).T
        
        # 🔍 驗證標準化效果（可選的調試信息）
        if len(data_segments) == 0:  # 只在第一個 segment 時顯示
            print(f"   📊 標準化範圍驗證 (第一個segment):")
            print(f"      ECG: [{ecg_normalized.min():.3f}, {ecg_normalized.max():.3f}]")
            print(f"      RA:  [{RA_normalized.min():.3f}, {RA_normalized.max():.3f}]")
            print(f"      RRI: [{RRI_normalized.min():.3f}, {RRI_normalized.max():.3f}]")
            print(f"      RRID:[{RRID_normalized.min():.3f}, {RRID_normalized.max():.3f}]")
        
        # Add record_id to the returned tuple
        data_segments.append((seg_4ch, seg_lbl, record_id))

    print(f"   完成處理，輸出 {len(data_segments)} 個數據段")
    return data_segments

# =====================
# PART G: