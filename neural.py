# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:06:16 2024

@author: JAGU
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import ast
import numpy as np

# CSV dosyasını oku
df = pd.read_csv('sound_dataset_with_all_features_and_segments.csv')

# Özellikleri liste formatından dönüştür
def convert_string_to_list(string):
    try:
        # Veriyi listeye dönüştür
        return np.array(ast.literal_eval(string))
    except (ValueError, SyntaxError):
        return np.zeros(26)  # Varsayılan bir dizi döndür

# Üçüncü sütundan itibaren özellikler ve etiketleri ayır
# apply kullanarak dönüşüm işlemi
X = df.iloc[:, 2:].apply(lambda col: col.map(lambda x: convert_string_to_list(x) if isinstance(x, str) else np.zeros(26)))
X = np.array([np.concatenate(row) for row in X.values])  # Özellikleri numpy dizisine dönüştür

# Etiketleri ayır
y = df.iloc[:, 1].values
y = y.astype(int)  # Etiketleri int türüne dönüştür

# Eğitim ve test verilerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Yapay Sinir Ağı Modelini oluştur
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Tahminler yap
y_pred = model.predict(X_test)

# Sonuçları değerlendirin
print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

