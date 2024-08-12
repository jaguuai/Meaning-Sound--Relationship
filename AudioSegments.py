# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:42:07 2024

@author: JAGU
"""

import numpy as np
import pandas as pd
import librosa

def calculate_pitch(y, sr):
    pitches, magnitudes = librosa.core.pitch.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    return np.mean(pitch) if len(pitch) > 0 else 0

def calculate_pitch_stability(pitch_values):
    return np.std(pitch_values)

def calculate_noise_ratio(y):
    signal_power = np.mean(y**2)
    noise_power = np.mean((y - np.mean(y))**2)
    return 10 * np.log10(signal_power / noise_power)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    # n_fft ve hop_length değerlerini ses dosyasının uzunluğuna göre ayarla
    if len(y) < 2048:
        n_fft = len(y)
        hop_length = len(y) // 4
    else:
        n_fft = 2048
        hop_length = 512
    
    segment_duration = 0.5  # 0.5 saniye
    frame_length = int(segment_duration * sr)
    
    if len(y) < frame_length:
        raise ValueError(f"Ses dosyası ({file_path}) segmentlenmek için çok kısa.")
    
    segments = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    segment_features = [calculate_pitch(segment, sr) for segment in segments]
    
    # Diğer özellikler
    pitch = calculate_pitch(y, sr)
    pitch_stability = calculate_pitch_stability(segment_features)
    noise_ratio = calculate_noise_ratio(y)
    
    return pitch, pitch_stability, noise_ratio

# Mevcut CSV dosyasını yükleyin
df = pd.read_csv('sound_dataset_with_all_features_and_time_frequency.csv')

# Yeni özellikleri hesaplayın ve ekleyin
pitches = []
pitch_stabilities = []
noise_ratios = []

for file_path in df['File Path']:
    pitch, pitch_stability, noise_ratio = extract_features(file_path)
    pitches.append(pitch)
    pitch_stabilities.append(pitch_stability)
    noise_ratios.append(noise_ratio)

df['Pitch'] = pitches
df['Pitch Stability'] = pitch_stabilities
df['Noise Ratio'] = noise_ratios

# Güncellenmiş CSV dosyasını kaydedin
df.to_csv('sound_dataset_with_all_features_END.csv', index=False)

print("Ses segmentleri ve diğer özellikler başarıyla eklendi.")
