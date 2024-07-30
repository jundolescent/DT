import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import os
import neurokit2 as nk
# import cv2
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import resample

def load_raw_data(df, sampling_rate, path, num_samples=None):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr[:num_samples]]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr[:num_samples]]
    data = np.array([signal for signal, meta in data])
    return data


def save_ecg_as_image(X, path, sampling_rate=100):
    if not os.path.exists(path):
        os.makedirs(path)
        
    # for comparison, we only use 4096 samples of the signal    
    for i, signal in enumerate(X):
        time_axis = np.arange(signal.shape[1]) / sampling_rate
        normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        plt.figure(figsize=(10,4))
        plt.plot(time_axis, normalized_signal.T, color='gray')
        # plt.title(f'ECG Signal {i+1}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        plt.savefig(os.path.join(path, f'ecg_{i+1}.png'))
        plt.close()
        
        
path = '/home/work/jslee/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
sampling_rate=500
ecg_list = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# ECG data format
# 00001_lr.dat 16 1000.0(0)/mV 16 0 -119 1508 0 I
# 00001_lr.dat 16 1000.0(0)/mV 16 0 -55 723 0 II
# 00001_lr.dat 16 1000.0(0)/mV 16 0 64 64758 0 III
# 00001_lr.dat 16 1000.0(0)/mV 16 0 86 64423 0 AVR
# 00001_lr.dat 16 1000.0(0)/mV 16 0 -91 1211 0 AVL
# 00001_lr.dat 16 1000.0(0)/mV 16 0 4 7 0 AVF
# 00001_lr.dat 16 1000.0(0)/mV 16 0 -69 63827 0 V1
# 00001_lr.dat 16 1000.0(0)/mV 16 0 -31 6999 0 V2
# 00001_lr.dat 16 1000.0(0)/mV 16 0 0 63759 0 V3
# 00001_lr.dat 16 1000.0(0)/mV 16 0 -26 61447 0 V4
# 00001_lr.dat 16 1000.0(0)/mV 16 0 -39 64979 0 V5
# 00001_lr.dat 16 1000.0(0)/mV 16 0 -79 832 0 V6

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path, num_samples=20000)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# # Split data into train and test
# test_fold = 10
# # Train
# X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# # Test
# X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

image_save_path = f'/home/work/jslee/data/ecg_images/records{sampling_rate}_v2'

def _transpose_data(data):
    return np.transpose(data)

def transpose_data(data):
    return np.array([_transpose_data(signal) for signal in data])

transposed = transpose_data(X)

scaler = MinMaxScaler(feature_range=(-1, 1))
start_index = 904 if sampling_rate == 500 else 0
num_samples = 4096 if sampling_rate == 500 else 1000
target_num_samples = 512 if sampling_rate == 500 else 1000

for index, d in enumerate(transposed):
    folder_path = os.path.join(image_save_path, str(index))
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        pass
    for index2, signal in enumerate(d):
        selected_signal = signal[start_index:start_index + num_samples]
        time_axis = np.arange(selected_signal.shape[0]) / sampling_rate
        # Apply bandpass filter
        filtered_signal = nk.signal_filter(selected_signal, lowcut=0.05, highcut=150, sampling_rate=sampling_rate, method='butterworth', order=4)
        downsampled_signal = resample(filtered_signal, target_num_samples)
        # Reshape signal for scaler
        signal_reshaped = filtered_signal.reshape(-1, 1)
        normalized_signal = scaler.fit_transform(signal_reshaped).flatten()
        plt.figure(figsize=(10,4))
        plt.plot(time_axis, normalized_signal.T, color='gray')
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1]) 
        # plt.title(f'ECG Signal {index+1}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        image_path = os.path.join(image_save_path, f'{index}/ecg_{ecg_list[index2]}.png')
        if not os.path.exists(image_path):
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()



# To-do: generalize for all records
# ecg_signal = transposed[0][0]

# peaks, info = nk.ecg_peaks(ecg_signal, sampling_rate=100)
# # Extract clean EDA and SCR features
# hrv_time = nk.hrv_time(peaks, sampling_rate=100, show=True)
# print('hrv_time:', hrv_time)

# rpeaks = info['ECG_R_Peaks']
# # Delineate the ECG signal
# _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=100, method="peak")

# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
# plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:3], 
#                        waves_peak['ECG_P_Peaks'][:3],
#                        waves_peak['ECG_Q_Peaks'][:3],
#                        waves_peak['ECG_S_Peaks'][:3]], ecg_signal[:4000])

# Plot the ECG signal with delineated peaks
# print(waves_peak)
# n_peaks = len(waves_peak['ECG_T_Peaks']) - 1
# print(n_peaks)


# plt.figure(figsize=(10,4))
# plt.plot(ecg_signal)
# plt.scatter(rpeaks, ecg_signal[rpeaks], color='red', label='R-Peaks')
# plt.scatter(waves_peak['ECG_T_Peaks'][:n_peaks], ecg_signal[waves_peak['ECG_T_Peaks'][:n_peaks]], color='green') # , label='T-Peaks'
# plt.scatter(waves_peak['ECG_P_Peaks'][:n_peaks], ecg_signal[waves_peak['ECG_P_Peaks'][:n_peaks]], color='blue') # , label='P-Peaks'
# plt.scatter(waves_peak['ECG_Q_Peaks'][:n_peaks], ecg_signal[waves_peak['ECG_Q_Peaks'][:n_peaks]], color='orange') # , label='Q-Peaks'
# plt.scatter(waves_peak['ECG_S_Peaks'][:n_peaks], ecg_signal[waves_peak['ECG_S_Peaks'][:n_peaks]], color='purple') # , label='S-Peaks'
# plt.legend()
# plt.title('ECG Signal with Delineated Peaks')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.savefig(os.path.join(image_save_path, '0/ecg_Temp.png'))
# save_ecg_as_image(X[:2][0], image_save_path, sampling_rate)
# save_ecg_as_image(X[:2][1], image_save_path, sampling_rate)

# Visualize for test
# sample_signal = X_train[0]
# time_axis = np.arange(sample_signal.shape[1]) / sampling_rate
# plt.figure(figsize=(10,4))
# plt.plot(time_axis, sample_signal.T)
# plt.title('ECG Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()