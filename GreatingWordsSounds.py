# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 00:08:18 2024

@author: JAGU
"""

import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft


data = {
    "Greeting Words": {
        "en-AU": ["G’day", "Hello", "Hi", "How’s it going?", "How are ya?", "Good morning", "Good arvo", "Good evening", "Hey mate"],
        "en-US": ["Hello", "Hi", "Hey", "How’s it going?", "What’s up?", "Good morning", "Good afternoon", "Good evening", "Howdy", "Yo"],
        "en-GB": ["Hello", "Hi", "Hey", "How are you?", "How’s it going?", "Good morning", "Good afternoon", "Good evening"],
        "ko-KR": ["안녕하세요", "안녕", "여보세요", "좋은 아침", "좋은 오후", "좋은 저녁 ", "반가워요", "잘 지냈어요?", "어떻게 지내세요?", "뭐해요?"],
        "de-DE": ["Hallo", "Guten Morgen", "Guten Tag", "Guten Abend", "Guten Nacht", "Hi", "Servus", "Grüß Gott", "Wie geht’s?", "Was gibt’s?"],
        "es-LA": ["Hola", "Buenos días", "Buenas tardes", "Buenas noches", "Cómo estás?", "Qué tal?", "Qué pasa?", "Hola, ¿cómo te va?", "Cómo te va?", "Qué onda?"],
        "es-US": ["Hola", "Buenos días", "Buenas tardes", "Buenas noches", "Cómo estás?", "Qué tal?", "Qué pasa?", "Hola, ¿cómo te va?", "Cómo te va?", "Qué onda?"],
        "fr-FR": ["Bonjour", "Salut", "Bonsoir", "Bonne nuit", "Coucou", "Comment ça va?", "Ça va?", "Quoi de neuf?", "Comment allez-vous?", "Allô"],
        "fr-CA": ["Bonjour", "Salut", "Bonsoir", "Bonne nuit", "Coucou", "Comment ça va?", "Ça va?", "Quoi de neuf?", "Comment allez-vous?", "Allô"],
        "ja-JP": ["こんにちは", "こんばんは", "おはようございます", "おやすみなさい", "こんにちは", "どうも", "お元気ですか？", "よろしくお願いします", "お疲れ様です", "いらっしゃいませ"],
        "it-IT": ["Ciao", "Buongiorno", "Buonasera", "Buonanotte", "Salve", "Come stai?", "Come va?", "Che fai?", "Saluti", "Ehi"],
        "pt-BR": ["Olá", "Oi", "Bom dia", "Boa tarde", "Boa noite", "Como vai?", "Tudo bem?", "E aí?", "O que há de novo?", "Alô"],
        "nl-NL": ["Hallo", "Goedemorgen", "Goedemiddag", "Goedenavond", "Welterusten", "Hoi", "Hey", "Hoe gaat het?", "Alles goed?", "Dag"]
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
api_key="rdcHNkNEt_DfTIok6R7cv54mn5-yuhSQ42IAiLtZ-8wW"
url="https://api.au-syd.text-to-speech.watson.cloud.ibm.com/instances/5bf6bed1-0636-48c7-95e4-a45c87fba1c2"


from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator=IAMAuthenticator(api_key)
tts=TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(url)

import os
# Klasör yolu ve ses dosyalarını saklayacağınız dizi
output_dir = 'Sounds'  # Ana dizin
os.makedirs(output_dir, exist_ok=True)  # Klasörü oluştur, varsa hata vermez
# Geçersiz karakterleri temizleme işlevi
import re
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
def save_audio_files(data):
    for lang_code, words in data["Greeting Words"].items():
        voices = data["Voices"].get(lang_code, [])
        language_dir = os.path.join(output_dir, lang_code)
        os.makedirs(language_dir, exist_ok=True)
        
        for word_index, word in enumerate(words):
            sanitized_word = sanitize_filename(word)
            # Ses listesinin uzunluğu kadar döngü oluşturuluyor
            for voice_index, voice in enumerate(voices):
                audio_file_path = os.path.join(language_dir, f"{sanitized_word}_{voice}.wav")
                try:
                    response = tts.synthesize(
                        text=word,
                        voice=voice,
                        accept='audio/wav'
                    ).get_result()
                    
                    print(f"Response status code for '{word}' with voice '{voice}': {response.status_code}")
                    if response.status_code == 200:
                        if response.content:
                            with open(audio_file_path, 'wb') as audio_file:
                                audio_file.write(response.content)
                            print(f"Audio file created: {audio_file_path}")
                        else:
                            print(f"No audio content for '{word}' with voice '{voice}'.")
                    else:
                        print(f"Failed to get audio for '{word}' with voice '{voice}'. Status code: {response.status_code}")
                except Exception as e:
                    print(f"Exception occurred for '{word}' with voice '{voice}': {e}")

# Ses dosyalarını kaydetme
save_audio_files(data)