# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:59:31 2024

@author: JAGU
"""
import os
import pandas as pd
import librosa
from scipy.io import wavfile

# Klasör yolları
all_sounds_folder = 'all_sounds'
all_another_sounds_folder = 'all_another_words_sounds'

def get_waveform_librosa(file_path):
    """Librosa kullanarak mono dalga formunu alır."""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def get_waveform_scipy(file_path):
    """Scipy kullanarak mono dalga formunu alır."""
    sr, y = wavfile.read(file_path)
    if len(y.shape) > 1:  # Stereo ise, mono'ya dönüştür
        y = y.mean(axis=1)
    return y, sr

# Veri seti için listeler
file_paths = []
labels = []

# all_sounds klasöründen gelen dosyalar için
for root, dirs, files in os.walk(all_sounds_folder):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            y, sr = get_waveform_librosa(file_path)  # veya get_waveform_scipy(file_path)
            file_paths.append(file_path)
            labels.append(1)  # all_sounds klasöründen gelen dosyalar için 1

# all_another_sounds klasöründen gelen dosyalar için
for root, dirs, files in os.walk(all_another_sounds_folder):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            y, sr = get_waveform_librosa(file_path)  # veya get_waveform_scipy(file_path)
            file_paths.append(file_path)
            labels.append(0)  # all_another_sounds klasöründen gelen dosyalar için 0

# Veri setini oluştur
df = pd.DataFrame({
    'File Path': file_paths,
    'Label': labels
})

# Veri setini karıştır
df = df.sample(frac=1).reset_index(drop=True)

# Veri setini bir CSV dosyasına kaydet
df.to_csv('sound_dataset_shuffled.csv', index=False)

print("Karışık veri seti başarıyla oluşturuldu ve 'sound_dataset_shuffled.csv' dosyasına kaydedildi.")


# # Frekans Alanı Özellikleri (Frequency-Domain Features)
# from scipy.stats import linregress

# def get_waveform_librosa(file_path):
#     """Librosa kullanarak mono dalga formunu alır."""
#     y, sr = librosa.load(file_path, sr=None, mono=True)
#     return y, sr


# def extract_frequency_domain_features(y, sr):
#     """Dalga formundan frekans alanı özelliklerini çıkarır."""
#     # Temel Frekans (Fundamental Frequency)
#     f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
#     fundamental_frequency = np.nanmean(f0) if f0 is not None else 0
    
#     # Spektrum (Spectrum)
#     spectrum = np.abs(librosa.stft(y))
    
#     # Spektral Enerji (Spectral Energy)
#     spectral_energy = np.sum(np.square(spectrum))
    
#     # Spektral Yoğunluk (Spectral Density)
#     spectral_density = np.mean(spectrum)
    
#     # Spektral Düzgünlük (Spectral Flatness)
#     spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
#     # Spektral Eğiklik (Spectral Slope)
#     frequencies = np.linspace(0, sr / 2, spectrum.shape[0])
#     mean_slope = np.mean([linregress(frequencies, spectrum[:,i])[0] for i in range(spectrum.shape[1])])
    
#     # Harmonik Oran (Harmonic Ratio)
#     harmonic_ratio = np.mean(librosa.feature.spectral_centroid(y=y) / (np.mean(spectrum) + 1e-6))
    
#     return {
#         'Fundamental Frequency': fundamental_frequency,
#         'Spectral Energy': spectral_energy,
#         'Spectral Density': spectral_density,
#         'Spectral Flatness': spectral_flatness,
#         'Spectral Slope': mean_slope,
#         'Harmonic Ratio': harmonic_ratio
#     }

# # Veri seti için listeler
# file_paths = []
# labels = []
# features_list_time = []
# features_list_freq = []

# # all_sounds klasöründen gelen dosyalar için
# for root, dirs, files in os.walk(all_sounds_folder):
#     for file in files:
#         if file.endswith('.wav'):
#             file_path = os.path.join(root, file)
#             y, sr = get_waveform_librosa(file_path)
#             features_time = extract_time_domain_features(y, sr)
#             features_freq = extract_frequency_domain_features(y, sr)
#             file_paths.append(file_path)
#             labels.append(1)
#             features_list_time.append(features_time)
#             features_list_freq.append(features_freq)

# # all_another_sounds klasöründen gelen dosyalar için
# for root, dirs, files in os.walk(all_another_sounds_folder):
#     for file in files:
#         if file.endswith('.wav'):
#             file_path = os.path.join(root, file)
#             y, sr = get_waveform_librosa(file_path)
#             features_time = extract_time_domain_features(y, sr)
#             features_freq = extract_frequency_domain_features(y, sr)
#             file_paths.append(file_path)
#             labels.append(0)
#             features_list_time.append(features_time)
#             features_list_freq.append(features_freq)

# # Veri setini oluştur
# df = pd.DataFrame({
#     'File Path': file_paths,
#     'Label': labels
# })

# # Zaman alanı özelliklerini DataFrame'e ekle
# features_df_time = pd.DataFrame(features_list_time)
# df = pd.concat([df, features_df_time], axis=1)

# # Frekans alanı özelliklerini DataFrame'e ekle
# features_df_freq = pd.DataFrame(features_list_freq)
# df = pd.concat([df, features_df_freq], axis=1)

# # Veri setini karıştır
# df = df.sample(frac=1).reset_index(drop=True)

# # Veri setini bir CSV dosyasına kaydet
# csv_file = 'sound_dataset_with_all_features.csv'
# df.to_csv(csv_file, index=False)

# print(f"Dalga formu ve frekans alanı özellikleri başarıyla eklendi ve '{csv_file}' dosyasına kaydedildi.")



# # Cepstral Özellikler (Cepstral Features)

# def extract_cepstral_features(y, sr):
#     """Dalga formundan cepstral özellikleri çıkarır."""
#     # Mel-Frekans Cepstral Katsayıları (MFCCs)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfccs_mean = np.mean(mfccs, axis=1)  # Ortalama MFCC'ler

#     # Power Spectrum Cepstrum
#     S = np.abs(librosa.stft(y))**2
#     log_spectrum = np.log1p(S)
#     power_spectrum_cepstrum = np.fft.ifft(log_spectrum, axis=0)
#     power_spectrum_cepstrum_mean = np.mean(np.abs(power_spectrum_cepstrum), axis=1)

#     return {
#         'MFCCs': mfccs_mean.tolist(),
#         'Power Spectrum Cepstrum': power_spectrum_cepstrum_mean.tolist()
#     }

# # Özellikleri ekleyeceğimiz veri seti dosyasının yolu
# csv_file = 'sound_dataset_with_all_features.csv'

# # Veri setini yükle
# df = pd.read_csv(csv_file)

# # Cepstral özellikler için listeler
# mfccs_list = []
# power_spectrum_cepstrum_list = []

# # Özellikleri hesapla ve listeye ekle
# for file_path in df['File Path']:
#     y, sr = get_waveform_librosa(file_path)
#     features = extract_cepstral_features(y, sr)
#     mfccs_list.append(features['MFCCs'])
#     power_spectrum_cepstrum_list.append(features['Power Spectrum Cepstrum'])

# # Cepstral özellikleri veri çerçevesine ekle
# df['MFCCs'] = mfccs_list
# df['Power Spectrum Cepstrum'] = power_spectrum_cepstrum_list

# # Veri setini karıştır
# df = df.sample(frac=1).reset_index(drop=True)

# # Özelliklerle birlikte veri setini bir CSV dosyasına kaydet
# csv_file_with_cepstral_features = 'sound_dataset_with_all_features_and_cepstral.csv'
# df.to_csv(csv_file_with_cepstral_features, index=False)

# print(f"Cepstral özellikler başarıyla eklendi ve '{csv_file_with_cepstral_features}' dosyasına kaydedildi.")

# # İstatistiksel Özellikler (Statistical Features)
# from scipy.stats import skew, kurtosis

# def extract_statistical_features(y):
#     """Dalga formundan istatistiksel özellikleri çıkarır."""
#     # Ortalama (Mean)
#     mean = np.mean(y)
    
#     # Standart Sapma (Standard Deviation)
#     std_dev = np.std(y)
    
#     # Skewness (Asimetrik)
#     skewness = skew(y)
    
#     # Kurtosis (Sivrilik)
#     kurt = kurtosis(y)
    
#     return {
#         'Mean': mean,
#         'Standard Deviation': std_dev,
#         'Skewness': skewness,
#         'Kurtosis': kurt
#     }

# # Özellikleri ekleyeceğimiz veri seti dosyasının yolu
# csv_file = 'sound_dataset_with_all_features_and_cepstral.csv'

# # Veri setini yükle
# df = pd.read_csv(csv_file)

# # İstatistiksel özellikler için listeler
# mean_list = []
# std_dev_list = []
# skewness_list = []
# kurtosis_list = []

# # Özellikleri hesapla ve listeye ekle
# for file_path in df['File Path']:
#     y, sr = get_waveform_librosa(file_path)
#     stats = extract_statistical_features(y)
#     mean_list.append(stats['Mean'])
#     std_dev_list.append(stats['Standard Deviation'])
#     skewness_list.append(stats['Skewness'])
#     kurtosis_list.append(stats['Kurtosis'])

# # İstatistiksel özellikleri veri çerçevesine ekle
# df['Mean'] = mean_list
# df['Standard Deviation'] = std_dev_list
# df['Skewness'] = skewness_list
# df['Kurtosis'] = kurtosis_list

# # Veri setini karıştır
# df = df.sample(frac=1).reset_index(drop=True)

# # Özelliklerle birlikte veri setini bir CSV dosyasına kaydet
# csv_file_with_statistical_features = 'sound_dataset_with_all_featuresl.csv'
# df.to_csv(csv_file_with_statistical_features, index=False)

# print(f"İstatistiksel özellikler başarıyla eklendi ve '{csv_file_with_statistical_features}' dosyasına kaydedildi.")












