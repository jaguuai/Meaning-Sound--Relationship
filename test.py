# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 20:36:54 2024

@author:JAGU

"""

# authenticate
api_key="rdcHNkNEt_DfTIok6R7cv54mn5-yuhSQ42IAiLtZ-8wW"
url="https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/5bf6bed1-0636-48c7-95e4-a45c87fba1c2"


from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator=IAMAuthenticator(api_key)
tts=TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(url)
# Get the list of voices
voices = tts.list_voices().get_result()
for voice in voices['voices']:
    # print(f"Name: {voice['name']}, Language: {voice['language']}, Gender: {voice['gender']}")
    print(voice['language'])
# İngilizce metni sese dönüştürme
en_text = "Merhaba"  # İngilizce metin
with open('merhaba.mp3', 'wb') as audio_file:
    response = tts.synthesize(en_text, accept='audio/mp3', voice='en-US_LisaExpressive').get_result()
    audio_file.write(response.content)

from pydub import AudioSegment

# FFmpeg'in yolunu belirtin (FFmpeg'in yüklü olduğu dizine göre değiştirin)
AudioSegment.converter = "C:/ffmpeg/bin/ffmpeg.exe"

# MP3 dosyasını WAV formatına dönüştür
audio = AudioSegment.from_mp3('See you later.mp3')
audio.export('See you later.wav', format='wav')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft
import os

# WAV dosyalarını listele
wav_files = [f for f in os.listdir() if f.endswith('.wav')]

for wav_file in wav_files:
    # WAV dosyasını oku
    sample_rate, data = wavfile.read(wav_file)

    # Eğer stereo ise, sadece bir kanalı al
    if len(data.shape) > 1:
        data = data[:, 0]

    # Fourier dönüşümünü uygula
    N = len(data)
    yf = fft(data)
    xf = np.fft.fftfreq(N, 1 / sample_rate)

    # Pozitif frekansları seç
    positive_freqs = xf[:N // 2]
    positive_spectrum = np.abs(yf[:N // 2])

    # En yüksek genliğe sahip frekansı bul
    max_index = np.argmax(positive_spectrum)
    fundamental_freq = positive_freqs[max_index]

    # Sonucu yazdır
    print(f'{wav_file} Fundamental Frequency: {fundamental_freq:.2f} Hz')

    # # Grafik üzerinde göster
    # plt.figure(figsize=(10, 5))
    # plt.plot(positive_freqs, positive_spectrum)
    # plt.title(f'Frequency Spectrum of {wav_file}')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()

# Ayarlar
sample_rate =  22050  # Hz
frequency =280     # Hz
duration = 2         # saniye

# Zaman dizisi
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Sinüs dalgası oluştur
wave = 0.5 * np.sin(2 * np.pi * frequency * t)

# WAV dosyasına yaz
wavfile.write('280Hz_tone.wav', sample_rate, wave.astype(np.float32))









