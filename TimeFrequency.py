# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:32:55 2024

@author: JAGU
"""

import pandas as pd
import numpy as np
import librosa
import pywt

# Mevcut CSV dosyasını oku
df = pd.read_csv('sound_dataset_with_all_features_and_statistical.csv')

def get_waveform_librosa(file_path):
    """Librosa kullanarak mono dalga formunu alır."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def extract_time_frequency_features(y, sr):
    """Zaman-Frekans alanı özelliklerini çıkarır."""
    # STFT (Short-Time Fourier Transform)
    stft = np.abs(librosa.stft(y))
    stft_mean = np.mean(stft)
    stft_std = np.std(stft)

    # Wavelet Dönüşümü (Wavelet Transform)
    coeffs, freqs = pywt.cwt(y, np.arange(1, 129), 'morl')
    wavelet_mean = np.mean(coeffs)
    wavelet_std = np.std(coeffs)
    
    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma)
    chroma_std = np.std(chroma)

    return {
        'STFT Mean': stft_mean,
        'STFT Std': stft_std,
        'Wavelet Mean': wavelet_mean,
        'Wavelet Std': wavelet_std,
        'Chroma Mean': chroma_mean,
        'Chroma Std': chroma_std
    }

# Zaman-Frekans alanı özelliklerini depolamak için liste
features_list_time_freq = []

for file_path in df['File Path']:
    y, sr = get_waveform_librosa(file_path)
    features_time_freq = extract_time_frequency_features(y, sr)
    features_list_time_freq.append(features_time_freq)

# Özellikleri DataFrame'e ekle
features_df_time_freq = pd.DataFrame(features_list_time_freq)

# Orijinal DataFrame ile birleştir
df_with_all_features = pd.concat([df, features_df_time_freq], axis=1)

# Yeni CSV dosyasına kaydet
df_with_all_features.to_csv('sound_dataset_with_all_features_and_time_frequency.csv', index=False)

print("Zaman-Frekans alanı özellikleri başarıyla eklendi ve 'sound_dataset_with_all_features_and_time_frequency.csv' dosyasına kaydedildi.")
