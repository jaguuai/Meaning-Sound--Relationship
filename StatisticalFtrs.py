# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:30:34 2024

@author: JAGU
"""

import librosa
import numpy as np
import pandas as pd

# Ses dosyalarının bulunduğu klasör
sound_dataset_folder = 'sound_dataset_with_all_features_and_time_frequency.csv'  # Mevcut veri setinizin bulunduğu dosya

# Yeni özelliklerin ekleneceği DataFrame'i yükleyin
df = pd.read_csv(sound_dataset_folder)

def extract_features(file_path, segment_duration=1.0):
    """Ses dosyasından özellikler çıkarır."""
    y, sr = librosa.load(file_path, sr=None)
    
    # Eğer ses dosyası çok kısaysa segment oluşturma
    if len(y) < int(segment_duration * sr):
        print(f"File {file_path} is too short to be segmented.")
        return None, None, None

    # Ses segmentlerini çıkar
    segments = librosa.util.frame(y, frame_length=int(segment_duration * sr), hop_length=int(segment_duration * sr))
    
    # Örneğin, sadece ilk segmenti kullanarak pitch, pitch stability, noise ratio gibi özellikler hesaplanabilir
    pitch = librosa.core.piptrack(y=segments[:, 0], sr=sr)
    pitch_stability = np.std(pitch)
    noise_ratio = np.mean(np.abs(segments[:, 0])) / np.std(segments[:, 0])

    return pitch, pitch_stability, noise_ratio

# Özellikleri çıkaracağımız ses dosyalarının listesi (CSV'den alıyoruz)
file_paths = df['File Path'].tolist()

# Sonuçları depolamak için boş listeler
all_pitch = []
all_pitch_stability = []
all_noise_ratio = []

for file_path in file_paths:
    pitch, pitch_stability, noise_ratio = extract_features(file_path)
    if pitch is not None:
        all_pitch.append(np.mean(pitch))  # Ortalama pitch değerini ekleyelim
        all_pitch_stability.append(pitch_stability)
        all_noise_ratio.append(noise_ratio)
    else:
        all_pitch.append(None)
        all_pitch_stability.append(None)
        all_noise_ratio.append(None)

# Yeni özellikleri DataFrame'e ekleyelim
df['Pitch'] = all_pitch
df['Pitch Stability'] = all_pitch_stability
df['Noise Ratio'] = all_noise_ratio

# Yeni özellikleri içeren CSV dosyasını kaydedin
output_csv = 'sound_dataset_with_all_features_and_segments.csv'
df.to_csv(output_csv, index=False)

print(f"Yeni özellikler başarıyla eklendi ve '{output_csv}' dosyasına kaydedildi.")
