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
# PART C: Z-SCORE NORMALIZATION
# ============================

def zscore_normalize(ecg):
    mu = np.mean(ecg)
    sigma = np.std(ecg)
    if sigma < 1e-12:
        return ecg - mu
    return (ecg - mu) / sigma

# ============================
# PART D: R-PEAK DETECTION, AUX
# ============================

def detect_rpeaks(ecg, fs):
    import neurokit2 as nk
    # 使用 Hamilton 演算法偵測 R-peaks
    rpeaks, = nk.ecg_hamilton(ecg, sampling_rate=fs)
    # 修正 R-peaks 位置
    rpeaks, = nk.ecg_correct_rpeaks(ecg, rpeaks=rpeaks, sampling_rate=fs, tol=0.1)
    return rpeaks  # global indices of R-peaks

    # from wfdb import processing
    # xqrs = processing.XQRS(sig=ecg, fs=fs)
    # xqrs.detect()
    # return xqrs.qrs_inds  # global indices of R-peaks

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
    rec_path = os.path.join(base_dir, record_id)
    ecg_raw, fs = load_apnea_ecg_record(rec_path)

    # Only do wavelet denoising, remove zscore normalization
    ecg_denoised = wavelet_denoise(ecg_raw, "coif4", 6)

    ann_path = os.path.join(base_dir, record_id)
    try:
        labels = load_apnea_annotations(ann_path)
    except:
        labels = None

    # Use denoised ECG (not normalized) for r-peak detection
    rpeaks = detect_rpeaks(ecg_denoised, fs)

    data_segments = []
    segments = segment_ecg_with_context(ecg_denoised, fs, labels)
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

        # stack => shape (out_len, 4)
        seg_4ch = np.vstack([ecg_resamp, RA_resamp, RRI_resamp, RRID_resamp]).T
        # Add record_id to the returned tuple
        data_segments.append((seg_4ch, seg_lbl, record_id))

    return data_segments

# =====================
# PART G: RECORD-LEVEL SPLIT
# =====================

def process_segments_segment_level(base_dir, out_len=1024):
    """Process all segments and save with patient IDs"""
    output_dir = os.path.join(base_dir, "split_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # The 35 labeled records
    labeled_records = (
        [f"a{i:02d}" for i in range(1, 21)] +
        [f"b{i:02d}" for i in range(1, 6)] +
        [f"c{i:02d}" for i in range(1, 11)]
    )
    
    # test_records = [f"x{i:02d}" for i in range(1, 36)]
    
    print("Processing labeled records:", labeled_records)
    # print("Processing test records:", test_records)
    
    # Process labeled records (for train and val)
    all_segments = []
    for rec_id in labeled_records:
        print(f"Processing record: {rec_id}")
        segs = preprocess_record(rec_id, base_dir, out_len=out_len)
        all_segments.extend(segs)
        print(f"Processed {len(segs)} segments from {rec_id}")
    
    print(f"Total segments from labeled records: {len(all_segments)}")
    
    # # Process test records
    # test_segments = []
    # for rec_id in test_records:
    #     print(f"Processing test record: {rec_id}")
    #     segs = preprocess_record(rec_id, base_dir, out_len=out_len)
    #     test_segments.extend(segs)
    #     print(f"Processed {len(segs)} segments from {rec_id}")
    
    # print(f"Total segments from test records: {len(test_segments)}")
    
    # Shuffle and split labeled data into train/val
    random.shuffle(all_segments)
    n_total = len(all_segments)
    n_train = int(0.90 * n_total)
    
    # Split the data
    train_data = all_segments[:n_train]
    val_data = all_segments[n_train:]
    # test_data = test_segments
    
    # Convert to arrays and save
    trainX, trainY, trainIDs = to_xy(train_data)
    valX, valY, valIDs = to_xy(val_data)
    # testX, testY, testIDs = to_xy(test_data)
    
    # Save training data
    print(f"Saving training data: {len(train_data)} segments")
    np.save(os.path.join(output_dir, "trainX.npy"), trainX)
    np.save(os.path.join(output_dir, "trainY.npy"), trainY)
    np.save(os.path.join(output_dir, "trainIDs.npy"), trainIDs)
    
    # Save validation data
    print(f"Saving validation data: {len(val_data)} segments")
    np.save(os.path.join(output_dir, "valX.npy"), valX)
    np.save(os.path.join(output_dir, "valY.npy"), valY)
    np.save(os.path.join(output_dir, "valIDs.npy"), valIDs)
    
    # # Save test data
    # print(f"Saving test data: {len(test_data)} segments")
    # np.save(os.path.join(output_dir, "testX.npy"), testX)
    # np.save(os.path.join(output_dir, "testY.npy"), testY)
    # np.save(os.path.join(output_dir, "testIDs.npy"), testIDs)
    
    print(f"\nAll data saved in '{output_dir}':")
    print(f"Train: {len(train_data)} segments")
    print(f"Validation: {len(val_data)} segments")
    # print(f"Test: {len(test_data)} segments")

def to_xy(data_list):
    """Helper to convert list of tuples to numpy arrays"""
    if not data_list:
        return np.array([]), np.array([]), np.array([])
    
    try:
        # Each segment => shape (1024,4)
        X = np.stack([seg for seg, _, _ in data_list], axis=0)
        Y = np.array([lbl for _, lbl, _ in data_list], dtype=int)
        IDs = np.array([pid for _, _, pid in data_list])
        return X, Y, IDs
    except Exception as e:
        print(f"Error in to_xy conversion: {e}")
        raise

# =====================
# MAIN
# =====================

if __name__ == "__main__":
    base_dir = r"C:\python\apnea\apnea-ecg-database-1.0.0"  # path where a01.dat, a01.hea, etc. are located
    
    # You can run either or both:
    process_segments_segment_level(base_dir, out_len=1024)  # Generate train/val/test
