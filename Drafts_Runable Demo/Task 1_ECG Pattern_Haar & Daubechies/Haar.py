# Databricks notebook source
# Install required packages if not already installed
!pip install wfdb pywavelets matplotlib numpy scipy
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
import os
import urllib.request
import zipfile
# Set matplotlib to display plots inline
%matplotlib inline
# Set style for better visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100
# Check if the files exist, if not, download them
record_name = '100'
data_files = [f'{record_name}.dat', f'{record_name}.hea', f'{record_name}.atr']

if not all(os.path.exists(f) for f in data_files):
    print("Downloading record 100 from PhysioNet...")
    
    # Download the required files
    base_url = "https://physionet.org/static/published-projects/mitdb/"
    files_to_download = [
        f"{record_name}.dat",
        f"{record_name}.hea", 
        f"{record_name}.atr"
    ]
    
    for file in files_to_download:
        url = base_url + file
        try:
            urllib.request.urlretrieve(url, file)
            print(f"Downloaded {file}")
        except Exception as e:
            print(f"Error downloading {file}: {e}")
            
    # Try to import wfdb after download
    try:
        import wfdb
        record = wfdb.rdrecord(record_name)
        annotation = wfdb.rdann(record_name, 'atr')
    except ImportError:
        print("wfdb library not available. Creating synthetic ECG data for demonstration.")
        # Fall back to synthetic data if wfdb is not available
        record = None
else:
    print("Local files found. Loading from disk...")
    try:
        import wfdb
        record = wfdb.rdrecord(record_name)
        annotation = wfdb.rdann(record_name, 'atr')
    except ImportError:
        print("wfdb library not available. Creating synthetic ECG data for demonstration.")
        record = None

# If we couldn't load the real data, create a synthetic ECG signal for demonstration
if record is None:
    print("Generating synthetic ECG signal for demonstration...")
    fs = 360  # Sampling frequency (Hz)
    t = np.arange(0, 10, 1/fs)  # 10 seconds of data
    # Create a synthetic ECG signal
    ecg_signal = np.sin(2 * np.pi * 1 * t)  # Basic sinus rhythm
    # Add P wave
    ecg_signal += 0.2 * np.exp(-50 * (t % 1 - 0.2)**2) 
    # Add QRS complex
    ecg_signal += 1.5 * np.exp(-500 * (t % 1 - 0.4)**2)
    # Add T wave
    ecg_signal += 0.3 * np.exp(-50 * (t % 1 - 0.6)**2)
    # Add some noise
    np.random.seed(42)  # For reproducibility
    ecg_signal += 0.1 * np.random.randn(len(t))
else:
    # Extract the ECG signal and metadata from the real record
    ecg_signal = record.p_signal[:, 0]  # Using the first channel (MLII)
    fs = record.fs  # Sampling frequency (Hz)

signal_length = len(ecg_signal)
time = np.arange(signal_length) / fs

print(f"ECG Signal Length: {signal_length} samples")
print(f"Sampling Frequency: {fs} Hz")
print(f"Recording Duration: {signal_length/fs:.2f} seconds")
# Plot the raw ECG signal
plt.figure(figsize=(12, 6))
plt.plot(time, ecg_signal, color='b', linewidth=1)
plt.title('ECG Signal (Real or Synthetic for Demonstration)', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude (mV)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
# Perform DWT with Haar wavelet
# We'll use 4 levels of decomposition
coeffs = pywt.wavedec(ecg_signal, 'haar', level=4)

# Extract approximation and detail coefficients
cA4, cD4, cD3, cD2, cD1 = coeffs

print(f"Original signal length: {len(ecg_signal)}")
print(f"Approximation coefficients (cA4) length: {len(cA4)}")
print(f"Detail coefficients level 4 (cD4) length: {len(cD4)}")
print(f"Detail coefficients level 3 (cD3) length: {len(cD3)}")
print(f"Detail coefficients level 2 (cD2) length: {len(cD2)}")
print(f"Detail coefficients level 1 (cD1) length: {len(cD1)}")
# Create time arrays for each coefficient level
time_cA4 = np.linspace(0, signal_length/fs, len(cA4))
time_cD4 = np.linspace(0, signal_length/fs, len(cD4))
time_cD3 = np.linspace(0, signal_length/fs, len(cD3))
time_cD2 = np.linspace(0, signal_length/fs, len(cD2))
time_cD1 = np.linspace(0, signal_length/fs, len(cD1))

# Plot the coefficients
fig, axs = plt.subplots(5, 1, figsize=(12, 10))

axs[0].plot(time_cA4, cA4, color='r', linewidth=1)
axs[0].set_title('Approximation Coefficients (cA4)', fontsize=12)
axs[0].set_ylabel('Amplitude')
axs[0].grid(True)

axs[1].plot(time_cD4, cD4, color='g', linewidth=1)
axs[1].set_title('Detail Coefficients Level 4 (cD4)', fontsize=12)
axs[1].set_ylabel('Amplitude')
axs[1].grid(True)

axs[2].plot(time_cD3, cD3, color='m', linewidth=1)
axs[2].set_title('Detail Coefficients Level 3 (cD3)', fontsize=12)
axs[2].set_ylabel('Amplitude')
axs[2].grid(True)

axs[3].plot(time_cD2, cD2, color='c', linewidth=1)
axs[3].set_title('Detail Coefficients Level 2 (cD2)', fontsize=12)
axs[3].set_ylabel('Amplitude')
axs[3].grid(True)

axs[4].plot(time_cD1, cD1, color='b', linewidth=1)
axs[4].set_title('Detail Coefficients Level 1 (cD1)', fontsize=12)
axs[4].set_ylabel('Amplitude')
axs[4].set_xlabel('Time (seconds)')
axs[4].grid(True)

plt.tight_layout()
plt.suptitle('Wavelet Decomposition Coefficients (Haar Wavelet)', fontsize=14, y=1.02)
plt.show()
# Reconstruct the signal from wavelet coefficients
reconstructed_signal = pywt.waverec(coeffs, 'haar')

# Due to the DWT process, the reconstructed signal might have a slightly different length
# We'll trim it to match the original signal length
min_length = min(len(ecg_signal), len(reconstructed_signal))
ecg_signal_trimmed = ecg_signal[:min_length]
reconstructed_signal_trimmed = reconstructed_signal[:min_length]
time_trimmed = time[:min_length]

# Calculate the reconstruction error
reconstruction_error = np.abs(ecg_signal_trimmed - reconstructed_signal_trimmed)
# Plot original and reconstructed signals
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_trimmed, ecg_signal_trimmed, 'b-', linewidth=1, label='Original Signal')
plt.plot(time_trimmed, reconstructed_signal_trimmed, 'r--', linewidth=1, label='Reconstructed Signal')
plt.title('Original vs Reconstructed ECG Signal', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude (mV)', fontsize=12)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_trimmed, reconstruction_error, 'g-', linewidth=1)
plt.title('Reconstruction Error', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
# Define a function for wavelet denoising
def wavelet_denoising(signal, wavelet='haar', level=4, threshold_type='soft'):
    # Decompose the signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Calculate threshold (using universal threshold)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate noise standard deviation
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # Universal threshold
    
    # Apply threshold to detail coefficients
    coeffs_thresholded = [coeffs[0]]  # Keep approximation coefficients unchanged
    for i in range(1, len(coeffs)):
        if threshold_type == 'soft':
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        else:  # hard thresholding
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
    
    # Reconstruct the signal
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)
    
    return denoised_signal[:len(signal)]  # Ensure same length as input

# Apply denoising
denoised_signal = wavelet_denoising(ecg_signal, wavelet='haar', level=4, threshold_type='soft')

# Calculate the difference (noise removed)
noise_removed = ecg_signal - denoised_signal

# Plot the results
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time, ecg_signal, 'b-', linewidth=1)
plt.title('Original ECG Signal', fontsize=14)
plt.ylabel('Amplitude (mV)', fontsize=12)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, denoised_signal, 'r-', linewidth=1)
plt.title('Denoised ECG Signal (Wavelet Thresholding)', fontsize=14)
plt.ylabel('Amplitude (mV)', fontsize=12)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, noise_removed, 'g-', linewidth=1)
plt.title('Noise Removed', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude (mV)', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
# Calculate performance metrics
def calculate_snr(original, processed):
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - processed)**2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def calculate_rmse(original, processed):
    return np.sqrt(np.mean((original - processed)**2))

# Calculate metrics for reconstruction and denoising
snr_original = calculate_snr(ecg_signal, np.zeros_like(ecg_signal))  # Reference
snr_reconstructed = calculate_snr(ecg_signal_trimmed, reconstructed_signal_trimmed)
snr_denoised = calculate_snr(ecg_signal, denoised_signal)

rmse_reconstructed = calculate_rmse(ecg_signal_trimmed, reconstructed_signal_trimmed)
rmse_denoised = calculate_rmse(ecg_signal, denoised_signal)

print("Performance Metrics:")
print(f"SNR after Reconstruction: {snr_reconstructed:.2f} dB")
print(f"SNR after Denoising: {snr_denoised:.2f} dB")
print(f"RMSE after Denoising: {rmse_denoised:.6f}")
# Final summary
print("ANALYSIS SUMMARY")
print("================")
print("1. Successfully loaded/created and visualized ECG data")
print("2. Applied 4-level DWT decomposition using Haar wavelet")
print("3. Reconstructed signal with minimal error (near-perfect reconstruction)")
print("4. Implemented wavelet thresholding for denoising")
print("5. Demonstrated the potential of wavelet transforms for healthcare signal analysis")
print("\nThis analysis demonstrates the capability to perform meaningful healthcare signal")
print("analysis remotely using wavelet transforms, which can be valuable for remote")
print("patient monitoring and telemedicine applications.")