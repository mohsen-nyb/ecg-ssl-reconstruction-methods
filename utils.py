import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import glob
import random
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import csv
import scipy.stats as stats



def parse_tfr_element(element):
    data = {
        'ECG_LEAD_I': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_II': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_III': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_AVR': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_AVL': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_AVF': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V1': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V2': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V3': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V4': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V5': tf.io.FixedLenFeature([], tf.string),
        'ECG_LEAD_V6': tf.io.FixedLenFeature([], tf.string),
        'abnormal': tf.io.FixedLenFeature([], tf.float32)
    }

    content = tf.io.parse_single_example(element, data)

    def decode_and_normalize(tensor):
        signal = tf.cast(tf.io.parse_tensor(tensor, out_type=tf.int32)[:5000], tf.float32) / 4247.0
        return signal - tf.reduce_mean(signal)

    ecg_leads = [decode_and_normalize(content[key]) for key in sorted(data.keys()) if key != 'abnormal']
    signal = tf.stack(ecg_leads, axis=0)
    return signal.numpy(), content['abnormal'].numpy()



def set_seed(seed=42):
    """
    Set all random seeds to ensure reproducibility across runs.
    """
    torch.manual_seed(seed)  # For PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For CUDA (if using GPUs)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)  # For NumPy
    random.seed(seed)  # For Python's random module
    torch.backends.cudnn.deterministic = True  # Ensures deterministic results
    torch.backends.cudnn.benchmark = False  # Avoids random algorithms for performance optimization


class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels
        self.labels = np.array(self.labels)
        print('-----some data statistics-----')
        print(f'number of all samples: {len(self.labels)}')
        print(f'number samples with abnormal echo: {len(self.labels[self.labels==1])}')
        print(f'percentage of samples with abnormal echo: {len(self.labels[self.labels==1]) / len(self.labels)}')
        print()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label


def compute_metrics(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    auc_roc = roc_auc_score(y_true, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall_curve, precision_curve)
    return accuracy, precision, recall, f1, auc_roc, pr_auc

def record_metrics(phase, loss, preds, labels, metrics, epoch, loader_len):
    preds = np.array(preds)
    labels = np.array(labels)
    accuracy, precision, recall, f1, auc, pr_auc = compute_metrics(labels, preds)
    metrics[f'{phase}_loss'].append(loss / loader_len)
    # if phase =='train':
    #     metrics[f'cl_loss'].append(cl_loss / loader_len)
    metrics[f'{phase}_accuracy'].append(accuracy)
    metrics[f'{phase}_precision'].append(precision)
    metrics[f'{phase}_recall'].append(recall)
    metrics[f'{phase}_f1'].append(f1)
    metrics[f'{phase}_auc'].append(auc)
    metrics[f'{phase}_pr_auc'].append(pr_auc)

def plot_metrics(metrics_df, metric_name, model_name, save_folder):
    plt.figure(figsize=(10, 5))
    if metric_name == 'cl_loss':
        plt.plot(metrics_df['epoch_cl'], metrics_df['cl_loss'], label=f'Train {metric_name.capitalize()}')
    else:
        plt.plot(metrics_df['epoch'], metrics_df[f'train_{metric_name}'], label=f'Train {metric_name.capitalize()}')
        plt.plot(metrics_df['epoch'], metrics_df[f'val_{metric_name}'], label=f'Val {metric_name.capitalize()}')
        
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_folder, f'{metric_name}_plot.png'))
    plt.show()
    plt.close()

def save_metrics(metrics_df, experiment_name, save_folder):
    metrics_df.to_csv(os.path.join(save_folder, f'{experiment_name}_metrics.csv'), index=False)




def calculate_confidence_interval_error(performance_metrics_list, confidence_level=0.95):
    """
    Calculate the confidence interval and margin of error for a list of performance metrics.

    Parameters:
    performance_metrics_list (list): A list of performance metric values.
    confidence_level (float): The confidence level for the interval (default is 0.95 for 95%).

    Returns:
    tuple: A tuple containing the mean performance, the confidence interval (lower bound, upper bound),
           and the margin of error.
    """
    # Calculate mean and standard deviation
    mean_performance = np.mean(performance_metrics_list)
    std_performance = np.std(performance_metrics_list, ddof=1)  # Use ddof=1 for sample standard deviation

    # Number of samples
    n = len(performance_metrics_list)

    # Z-score for the given confidence level
    z = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate the margin of error
    margin_of_error = z * (std_performance / np.sqrt(n))

    # Calculate the confidence interval
    #confidence_interval = (mean_performance - margin_of_error, mean_performance + margin_of_error)

    return margin_of_error


    import torch

def ecg_to_frequency_domain(ecg_batch):
    """
    Convert an ECG batch from the time domain to the frequency domain using FFT.
    
    Args:
    - ecg_batch (torch.Tensor): A tensor of shape (batch_size, leads_num, signal_length)
    
    Returns:
    - freq_batch (torch.Tensor): The frequency domain representation of the ECG batch (complex values)
    - magnitude_batch (torch.Tensor): Magnitude of the frequency domain representation

    # Example usage:
        batch_size = 64
        leads = 12
        signal_length = 5000
        # Simulate a batch of ECG signals (random data for demonstration)
        ecg_batch = torch.randn(batch_size, leads, signal_length)

        # Convert to frequency domain
        freq_batch, magnitude_batch = ecg_to_frequency_domain(ecg_batch)

        # Verify the shapes of the output
        print(f"Frequency Domain Batch Shape: {freq_batch.shape}")
        print(f"Magnitude Batch Shape: {magnitude_batch.shape}")
    """
    # Perform FFT on the batch (along the signal_length dimension)
    freq_batch = torch.fft.fft(ecg_batch, dim=-1)
    
    # Compute the magnitude (absolute value of the FFT result)
    magnitude_batch = freq_batch.abs()
    
    return freq_batch, magnitude_batch



def plot_frequency_domain(magnitude_batch, signal_length, title="Magnitude Spectrum", index=0):
    """
    Plot the magnitude spectrum of a batch of ECG signals.
    
    Args:
    - magnitude_batch (torch.Tensor): The magnitude of the frequency domain representation.
    - signal_length (int): The length of the ECG signal.
    - title (str): Title of the plot.
    """
    # Create a frequency axis for the plot
    freqs = np.fft.fftfreq(signal_length)
    positive_freqs = freqs[:signal_length // 2]
    #f_s = 1000
    #freqs = np.fft.fftfreq(signal_length, d=1/f_s)
    
    # Plot the magnitude spectrum for the first few signals in the batch
    plt.plot(positive_freqs, magnitude_batch[index, 0, :signal_length // 2].cpu().numpy())  # For lead 0 (first lead)
    plt.title(f"{title} - Signal {index+1}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

def preprocess_magnitude_batch(magnitude_batch, method='zscore'):
    """
    Preprocess the magnitude batch using the specified method.
    
    Args:
    - magnitude_batch (torch.Tensor): The magnitude of the frequency domain representation.
    - method (str): The normalization method to apply ('zscore', 'minmax', 'log', etc.)
    
    Returns:
    - Preprocessed magnitude batch.
    """
    if method == 'zscore':
        # Z-score normalization (Standardization)
        mean = magnitude_batch.mean(dim=-1, keepdim=True)
        std = magnitude_batch.std(dim=-1, keepdim=True)
        magnitude_batch = (magnitude_batch - mean) / (std + 1e-6)  # Avoid division by zero
    
    elif method == 'minmax':
        # Min-Max scaling
        min_val = magnitude_batch.min(dim=-1, keepdim=True).values
        max_val = magnitude_batch.max(dim=-1, keepdim=True).values
        magnitude_batch = (magnitude_batch - min_val) / (max_val - min_val + 1e-6)  # Avoid division by zero
    
    elif method == 'log':
        # Log transformation
        magnitude_batch = torch.log(magnitude_batch + 1e-6)  # Adding small value to avoid log(0)
    
    return magnitude_batch


def set_seed(seed=42):
    """
    Set all random seeds to ensure reproducibility across runs.
    """
    torch.manual_seed(seed)  # For PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For CUDA (if using GPUs)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)  # For NumPy
    random.seed(seed)  # For Python's random module
    torch.backends.cudnn.deterministic = True  # Ensures deterministic results
    torch.backends.cudnn.benchmark = False  # Avoids random algorithms for performance optimization



def preprocessing(row, max_amp):
    data = {
        'ECG_LEAD_I': row['full_lead_I'],
        'ECG_LEAD_II': row['full_lead_II'],
        'ECG_LEAD_III': row['full_lead_III'],
        'ECG_LEAD_AVR': row['full_lead_AVR'],
        'ECG_LEAD_AVL': row['full_lead_AVL'],
        'ECG_LEAD_AVF': row['full_lead_AVF'],
        'ECG_LEAD_V1': row['full_lead_V1'],
        'ECG_LEAD_V2': row['full_lead_V2'],
        'ECG_LEAD_V3': row['full_lead_V3'],
        'ECG_LEAD_V4': row['full_lead_V4'],
        'ECG_LEAD_V5': row['full_lead_V5'],
        'ECG_LEAD_V6': row['full_lead_V6'],
        'label': row['Responder'],
        # 'mrn' : row['mrn']
    }

    def decode_and_normalize(arr, max_amp):
        signal = arr[:5000].astype(np.float32) / max_amp
        return signal - np.mean(signal)


    ecg_leads = np.stack([decode_and_normalize(data[key], max_amp) for key in sorted(data.keys()) if key != 'label'], axis=0)

    return ecg_leads, data['label']



def apply_mask(x, mask_ratio=0.5, mask_size=250):
    """
    x: (B, 12, 5000)
    Randomly masks continuous blocks of ECG data per channel.
    """
    B, C, L = x.shape
    num_masks = int((L * mask_ratio) // mask_size)
    mask = torch.ones_like(x)

    for b in range(B):
        for _ in range(num_masks):
            start = np.random.randint(0, L - mask_size)
            mask[b, :, start:start+mask_size] = 0
    
    return x * mask, mask
