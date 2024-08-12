# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:15:33 2024

@author: JAGU
"""

import pandas as pd
import numpy as np
import librosa
from scipy.fftpack import dct

# CSV dosyasını oku
df = pd.read_csv('sound_dataset_with_frequency_domain_features.csv')

def get_waveform_librosa(file_path):
    """Librosa kullanarak mono dalga formunu alır."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def extract_cepstral_features(y, sr):
    """Dalga formundan cepstral özellikleri çıkarır."""
    # Mel-Frekans Cepstral Katsayıları (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Power Spectrum Cepstrum
    # Spektrum
    spectrum = np.abs(librosa.stft(y))
    # Log spektral yoğunluk
    log_spectrum = np.log(np.mean(spectrum, axis=1) + 1e-6)
    # Invers Fourier dönüşümü
    power_spectrum_cepstrum = np.abs(dct(log_spectrum, type=2, norm='ortho'))
    
    return {
        'MFCCs Mean': mfccs_mean.tolist(),  # MFCCs ortalamasını liste olarak ekle
        'Power Spectrum Cepstrum': power_spectrum_cepstrum.tolist()  # Cepstrum'u liste olarak ekle
    }

# Cepstral özellikleri depolamak için liste
features_list_cepstral = []

for file_path in df['File Path']:
    y, sr = get_waveform_librosa(file_path)
    features_cepstral = extract_cepstral_features(y, sr)
    features_list_cepstral.append(features_cepstral)

# Özellikleri DataFrame'e ekle
features_df_cepstral = pd.DataFrame(features_list_cepstral)

# Orijinal DataFrame ile birleştir
df_with_features = pd.concat([df, features_df_cepstral], axis=1)

# Yeni CSV dosyasına kaydet
df_with_features.to_csv('sound_dataset_with_all_features.csv', index=False)

print("Cepstral özellikler başarıyla eklendi ve 'sound_dataset_with_all_features.csv' dosyasına kaydedildi.")
