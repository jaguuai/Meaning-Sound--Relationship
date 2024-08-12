# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 21:57:28 2024

@author: JAGU
"""

import pandas as pd
import numpy as np
import librosa

# CSV dosyasını oku
df = pd.read_csv('sound_dataset_shuffled.csv')

def get_waveform_librosa(file_path):
    """Librosa kullanarak mono dalga formunu alır."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def extract_time_domain_features(y, sr):
    """Dalga formundan zaman alanı özelliklerini çıkarır."""
    # Sinyal Enerjisi
    energy = np.sum(np.square(y)) / len(y)
    
    # Tepe Değeri
    peak_amplitude = np.max(np.abs(y))
    
    # Zero-Crossing Rate
    zero_crossings = np.sum(np.abs(np.diff(np.sign(y)))) / len(y)
    
    # RMS (Root Mean Square)
    rms = np.sqrt(np.mean(np.square(y)))
    
    # Temporal Centroid
    time = np.arange(len(y)) / sr
    temporal_centroid = np.sum(time * np.abs(y)) / np.sum(np.abs(y))
    
    return {
        'Energy': energy,
        'Peak Amplitude': peak_amplitude,
        'Zero-Crossing Rate': zero_crossings,
        'RMS': rms,
        'Temporal Centroid': temporal_centroid
    }

# Özellikleri depolamak için listeler
features_list = []

for file_path in df['File Path']:
    y, sr = get_waveform_librosa(file_path)
    features = extract_time_domain_features(y, sr)
    features_list.append(features)

# Özellikleri DataFrame'e ekle
features_df = pd.DataFrame(features_list)

# Orijinal DataFrame ile birleştir
df_with_features = pd.concat([df, features_df], axis=1)

# Yeni CSV dosyasına kaydet
df_with_features.to_csv('sound_dataset_with_time_domain_features.csv', index=False)

print("Zaman alanı özellikleri başarıyla eklendi ve 'sound_dataset_with_time_domain_features.csv' dosyasına kaydedildi.")