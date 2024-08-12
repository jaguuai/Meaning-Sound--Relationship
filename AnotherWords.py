# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 00:47:58 2024

@author: JAGU
"""

import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft


data = {
    "Greeting Words": {
        "en-AU": ["Let's grab a coffee.", "That's a ripper!", "How's the weather today?", "I'm heading to the beach.", "Did you catch the footy?","It's a scorcher outside.", "Can you pass the Vegemite?", "I'll see you at the barbie.", "It's time for brekky.","What's on the telly tonight?"],
        "en-US": ["What's up with the traffic?", "I need a coffee break.", "The game was intense!", "Let's meet for lunch.", "Have you seen the latest movie?", "I’m going for a run.", "The weather is amazing today.", "What's for dinner tonight?", "I'm working from home.", "Are you ready for the meeting?"],
        "en-GB": ["How's your day been?", "I fancy a cuppa.", "The train was late again.", "It's raining cats and dogs.", "Let's have a chat.", "I'm off to the shops.", "The match was thrilling.", "I'm heading to the pub."],
        "ko-KR": ["점심 뭐 먹을까? ", "저녁은 집에서 먹자.", "주말에 뭐 할 거야?", "어제 뉴스 봤어?", "책을 읽고 있어요.", "시간이 정말 빨리 가네요.", "고양이 너무 귀여워요.","오늘 날씨 어때요?"],
        "de-DE": ["Wie war dein Tag? ", "Ich brauche einen Kaffee.", "Das Wetter ist fantastisch.", "Lass uns einen Spaziergang machen.", "Hast du den Film gesehen? ", "Was gibt's zum Abendessen?", "Ich lese gerade ein Buch.", "Das Spiel war spannend. ", "Ich arbeite von zu Hause aus.?", "Wollen wir ins Kino gehen?"],
        "es-US": ["Cómo estuvo tu día?", "Necesito un café. ", "Vas a la reunión?", "El clima está perfecto.", "Vamos a cenar fuera.", "Viste la película?", "El partido fue increíble.", "Estoy leyendo un buen libro. ", "Qué haces este fin de semana?", "El tráfico es terrible."],
        "fr-FR": ["Comment s'est passée ta journée?", "J'ai besoin d'un café.", "Le film était incroyable.", "Quel temps fait-il aujourd'hui?", "Je vais faire du shopping.", "On se retrouve pour déjeuner?", "J'ai regardé la télé hier soir.", "Le match était intense. ", "Je prépare le dîner ce soir.", "Tu as vu les infos?"],
        "fr-CA": ["As-tu fini ton travail?", "Le hockey est mon sport préféré.", "On va au chalet ce week-end.", "Il fait froid aujourd'hui.", "As-tu vu la tempête? ", "Je vais prendre un café.", "Le film était vraiment bon.", "On se retrouve après le travail?", "J'ai fait une promenade hier soir.", "Le souper est prêt."],
        "ja-JP": ["今日はどうでしたか?", "映画を見に行こう", "明日は休みですか?", "天気がいいですね", "昼ご飯は何を食べますか?", "本を読んでいます", "公園に行きましょう", "これは美味しいですね", "買い物に行きます", "このドラマを見ましたか? "],
        "it-IT": ["Prendiamo un caffè? ", "Il film era bellissimo.", "Hai visto la partita?", "Sto leggendo un libro.", "Andiamo al parco.", "La cena è pronta.", "Che tempo fa oggi? ", "Vado a fare una passeggiata.", "Ci vediamo domani."],
        "pt-BR": ["Como foi seu dia?", "Vou tomar um café.", "O filme foi incrível. ", "Vamos ao parque hoje?", "Estou lendo um livro ótimo. ", "Qual é o plano para o jantar? ", "O jogo foi emocionante.", "Está muito quente hoje.", "Você viu as notícias? ", "Vamos almoçar fora?"],
        "nl-NL": ["Hoe was je dag?", "Ik heb koffie nodig. ", "De film was fantastisch. ", "Wat eten we vanavond?", "Het weer is vandaag mooi.", "Ik ga naar de supermarkt.", "Heb je de wedstrijd gezien?", "Ik lees momenteel een boek.", "Laten we een wandeling maken. ", "De trein was te laat."]
    },
    "Voices": {
        "en-AU": ["en-AU_HeidiExpressive", "en-AU_JackExpressive"],
        "en-US": ["en-US_MichaelExpressive", "en-US_LisaExpressive", "en-US_AllisonVoice", "en-US_AllisonExpressive", "en-US_EmmaExpressive", "en-US_MichaelVoice", "en-US_KevinV3Voice", "en-US_AllisonV3Voice", "en-US_AllisonV2Voice", "en-US_EmilyV3Voice"],
        "en-GB": ["en-GB_KateVoice", "en-GB_CharlotteV3Voice", "en-GB_JamesV3Voice"],
        "ko-KR": ["ko-KR_JinV3Voice"],
        "de-DE": ["de-DE_DieterVoice", "de-DE_BirgitVoice", "de-DE_DieterV3Voice", "de-DE_BirgitV3Voice", "de-DE_ErikaV3Voice", "de-DE_DieterV2Voice", "de-DE_BirgitV2Voice"],
        "es-LA": ["es-LA_LauraVoice", "es-LA_EnriqueVoice"],
        "es-US": ["es-US_SofiaVoice", "es-US_JorgeVoice"],
        "fr-FR": ["fr-FR_ReneeVoice", "fr-FR_NicolasV3Voice", "fr-FR_ReneeV3Voice"],
        "fr-CA": ["fr-CA_LouiseV3Voice"],
        "ja-JP": ["ja-JP_EmiV3Voice", "ja-JP_EmiVoice"],
        "it-IT": ["it-IT_FrancescaV2Voice", "it-IT_FrancescaVoice", "it-IT_FrancescaV3Voice"],
        "pt-BR": ["pt-BR_IsabelaV3Voice", "pt-BR_IsabelaVoice"],
        "nl-NL": ["nl-NL_MerelV3Voice"]
    },
    "Audio Files": []
}


# authenticate
api_key=""
url="https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/5bf6bed1-0636-48c7-95e4-a45c87fba1c2"


from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator=IAMAuthenticator(api_key)
tts=TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(url)

import os
# Klasör yolu ve ses dosyalarını saklayacağınız dizi
output_dir = 'AnotherWordsSounds'  # Ana dizin
os.makedirs(output_dir, exist_ok=True)  # Klasörü oluştur, varsa hata vermez
# Geçersiz karakterleri temizleme işlevi
import re
# def sanitize_filename(filename):
#     return re.sub(r'[<>:"/\\|?*]', '_', filename)
# def save_audio_files(data):
#     for lang_code, words in data["Greeting Words"].items():
#         voices = data["Voices"].get(lang_code, [])
#         language_dir = os.path.join(output_dir, lang_code)
#         os.makedirs(language_dir, exist_ok=True)
        
#         for word_index, word in enumerate(words):
#             sanitized_word = sanitize_filename(word)
#             # Ses listesinin uzunluğu kadar döngü oluşturuluyor
#             for voice_index, voice in enumerate(voices):
#                 audio_file_path = os.path.join(language_dir, f"{sanitized_word}_{voice}.wav")
#                 try:
#                     response = tts.synthesize(
#                         text=word,
#                         voice=voice,
#                         accept='audio/wav'
#                     ).get_result()
                    
#                     print(f"Response status code for '{word}' with voice '{voice}': {response.status_code}")
#                     if response.status_code == 200:
#                         if response.content:
#                             with open(audio_file_path, 'wb') as audio_file:
#                                 audio_file.write(response.content)
#                             print(f"Audio file created: {audio_file_path}")
#                         else:
#                             print(f"No audio content for '{word}' with voice '{voice}'.")
#                     else:
#                         print(f"Failed to get audio for '{word}' with voice '{voice}'. Status code: {response.status_code}")
#                 except Exception as e:
#                     print(f"Exception occurred for '{word}' with voice '{voice}': {e}")

# # Ses dosyalarını kaydetme
# save_audio_files(data)

import shutil

# Kaynak klasör (sounds klasörünüz)
source_folder = 'AnotherWordsSounds'

# Hedef klasör (Tüm wav dosyalarını toplamak için yeni bir klasör)
destination_folder = 'all_another_words_sounds'

# Hedef klasör yoksa oluştur
os.makedirs(destination_folder, exist_ok=True)

# Tüm alt klasörlerdeki wav dosyalarını topla
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.wav'):
            # Kaynak dosyanın tam yolu
            source_file_path = os.path.join(root, file)
            
            # Aynı dosya adı varsa üzerine yazmamak için dosya adı değiştirme (opsiyonel)
            new_file_name = f"{os.path.basename(root)}_{file}"
            destination_file_path = os.path.join(destination_folder, new_file_name)
            
            # Dosyayı hedef klasöre kopyala
            shutil.copy2(source_file_path, destination_file_path)
            print(f"Kopyalandı: {destination_file_path}")

print("Tüm dosyalar kopyalandı!")