# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:59:31 2024

@author: JAGU
"""

import os
import pandas as pd

# Klasör yolları
all_sounds_folder = 'all_sounds'
all_another_sounds_folder = 'all_another_words_sounds'

# Veri seti için listeler
file_paths = []
labels = []

# all_sounds klasöründen gelen dosyalar için
for root, dirs, files in os.walk(all_sounds_folder):
    for file in files:
        if file.endswith('.wav'):
            file_paths.append(os.path.join(root, file))
            labels.append(1)  # all_sounds klasöründen gelen dosyalar için 1

# all_another_sounds klasöründen gelen dosyalar için
for root, dirs, files in os.walk(all_another_sounds_folder):
    for file in files:
        if file.endswith('.wav'):
            file_paths.append(os.path.join(root, file))
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