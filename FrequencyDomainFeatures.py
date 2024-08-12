# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:05:39 2024

@author: JAGU
"""
import pandas as pd
import numpy as np
import librosa
from scipy.stats import linregress

# CSV dosyasını oku
df = pd.read_csv('sound_dataset_with_time_domain_features.csv')

def get_waveform_librosa(file_path):
    """Librosa kullanarak mono dalga formunu alır."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def extract_frequency_domain_features(y, sr):
    """Dalga formundan frekans alanı özelliklerini çıkarır."""
    # Temel Frekans (Fundamental Frequency)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    fundamental_frequency = np.nanmean(f0) if f0 is not None else 0
    
    # Spektrum (Spectrum)
    spectrum = np.abs(librosa.stft(y))
    
    # Spektral Enerji (Spectral Energy)
    spectral_energy = np.sum(np.square(spectrum))
    
    # Spektral Yoğunluk (Spectral Density)
    spectral_density = np.mean(spectrum)
    
    # Spektral Düzgünlük (Spectral Flatness)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # Spektral Eğiklik (Spectral Slope)
    frequencies = np.linspace(0, sr / 2, spectrum.shape[0])
    mean_slope = np.mean([linregress(frequencies, spectrum[:,i])[0] for i in range(spectrum.shape[1])])
    
    # Harmonik Oran (Harmonic Ratio)
    harmonic_ratio = np.mean(librosa.feature.spectral_centroid(y=y) / (np.mean(spectrum) + 1e-6))
    
    return {
        'Fundamental Frequency': fundamental_frequency,
        'Spectral Energy': spectral_energy,
        'Spectral Density': spectral_density,
        'Spectral Flatness': spectral_flatness,
        'Spectral Slope': mean_slope,
        'Harmonic Ratio': harmonic_ratio
    }

# Frekans alanı özelliklerini depolamak için liste
features_list_freq = []

for file_path in df['File Path']:
    y, sr = get_waveform_librosa(file_path)
    features_freq = extract_frequency_domain_features(y, sr)
    features_list_freq.append(features_freq)

# Özellikleri DataFrame'e ekle
features_df_freq = pd.DataFrame(features_list_freq)

# Orijinal DataFrame ile birleştir
df_with_features = pd.concat([df, features_df_freq], axis=1)

# Yeni CSV dosyasına kaydet
df_with_features.to_csv('sound_dataset_with_frequency_domain Featuresl_features.csv', index=False)

print("Frekans alanı özellikleri başarıyla eklendi ve 'sound_dataset_with_frequency_domain Featuresl_features.csv' dosyasına kaydedildi.")

