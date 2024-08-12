# Meaning-Sound-Relationship
Bu projede anne karnındaki bebeklerin konuşmayı ve dili bilmeden anlam çıkarması ve bazı müzıklerın bize sözsüz bir anlam ifade etmesi daha da genişletirssek seslerin hayvanlar ve tüm canlılar üzerinde ortak bir noktada yeri olabilir mi sorularının incelemek için başladım. 

Projede IBM Watson API Text-to-Speech free olarak yararlandım.
1-en-AU: Avustralya İngilizcesi
en-US: Amerikan İngilizcesi
en-GB: İngiltere İngilizcesi
2-ko-KR: Korece
3-de-DE: Almanca
4-es-US: Amerikan İspanyolcası
5-fr-FR: Fransızca (Fransa)
fr-CA: Kanada Fransızcası
6-ja-JP: Japonca
7-it-IT: İtalyanca
8-pt-BR: Brezilya Portekizcesi
9-nl-NL: Flemenkçe (Hollanda)
Burada görüldüğü üzere 9 farklı dil bulunmakta.Bazı diller bölgesel farklılıklarına göre ayrılmakta.
Ayrıca 

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

seklinde diller için kadın-erkek farklı ses seçenekleri bulunmakta .
Ben projemde selamlasma ifadelerini ayırt edici özellik olarak kullandım .Her dil için ifadelr buldum ve her ifadeyi tüm dil seçeneklerini döngüyle ses(wav)dosyasına IBM Text-to-speech yardımıyla dönüştürdü.->GreatingWordsSounds.py
Daha sonrasında işlemi selamlaşma harici oluşturduğum ifadeler setinede uyguladım.->AnotherWords.py
Bu ses dosyalarını tek klasörde birleştirdim ve selamlaşma kelimelerine label etiketi 1 diğerlerine 0 vererek csv dosyamı oluşturdum . Daha sonrasında bu verileri karıştırdım . (Eğitim kısmında daha verimli sonuç almak adına)->ShuffledDataset.py

Datasetime features eklemek için internetten sese ait özellikler hakkında bilgiler aldım.
1. Zaman Alanı Özellikleri (Time-Domain Features)
Dalga Formu (Waveform): Sesin zaman içindeki genlik değerleri.
Sinyal Enerjisi (Signal Energy): Sinyalin gücünün bir ölçüsü.
Tepe Değeri (Peak Amplitude): Sinyalin maksimum genlik değeri.
Zero-Crossing Rate: Sinyalin sıfır çizgisini geçiş sayısı; genellikle frekans içeriğinin bir göstergesidir.
Root Mean Square (RMS): Sinyalin enerji seviyesinin bir ölçüsü.
Temporal Centroid: Sesin ağırlıklı ortalama zaman merkezi.
2. Frekans Alanı Özellikleri (Frequency-Domain Features)
Temel Frekans (Fundamental Frequency): Sesin en düşük frekansı, genellikle sesin temel tonu.
Spektrum (Spectrum): Sesin frekans bileşenlerinin bir analizi.
Spektral Enerji (Spectral Energy): Frekans spektrumundaki toplam enerji.
Spektral Yoğunluk (Spectral Density): Frekans bandındaki enerji yoğunluğu.
Spektral Düzgünlük (Spectral Flatness): Spektrumun pürüzsüzlüğü, sesin ton kalitesini gösterir.
Spektral Eğiklik (Spectral Slope): Spektrumun eğimi, yüksek frekansların düşük frekanslara göre baskın olup olmadığını gösterir.
Harmonik Oran (Harmonic Ratio): Harmoniklerin toplam gücünün toplam spektral güce oranı.
3. Cepstral Özellikler (Cepstral Features)
Mel-Frekans Cepstral Katsayıları (MFCCs): Sesin insan kulağının algılayabileceği özelliklerini özetleyen en yaygın kullanılan ses özellikleri.
Power Spectrum Cepstrum: Sinyalin log spektral yoğunluğunun invers Fourier dönüşümü.
4. İstatistiksel Özellikler (Statistical Features)
Ortalama (Mean): Sinyal özelliklerinin ortalaması.
Standart Sapma (Standard Deviation): Sinyal özelliklerinin dağılımı.
Skewness: Sinyalin asimetrisi.
Kurtosis: Sinyalin sivriliği.
5. Zaman-Frekans Alanı Özellikleri (Time-Frequency Domain Features)
Short-Time Fourier Transform (STFT): Zaman ve frekans ekseninde değişen bir spektrum.
Wavelet Dönüşümü (Wavelet Transform): Zaman-frekans analizinde kullanılan çok çözünürlüklü analiz.
Chroma Features: Belirli bir anın spektrumunda belirli bir tonun baskınlığını gösterir.
6. Ses Segmentleri ve Diğer Özellikler
Ses Segmentleri (Audio Segments): Sinyali belirli süre aralıklarında bölebilirsiniz.
Ses Yüksekliği (Pitch): Sesin algılanan frekansı.
Ton Yüksekliği Stabilitesi (Pitch Stability): Tonun ne kadar sabit olduğu.
Gürültü Oranı (Noise Ratio): Sinyal içindeki gürültü seviyesi.

Daha sonrasında tüm bu ifadeeleri ayrı .py scriptlerinde uygun kütüphaneler ile çıkarıp csv dosyama ekledim.
1-TimeDomainFtrs.py->2-FrequencyDomainFeatures.py->3-CepstralFeatures.py->4-StatisticalFtrs.py->5-TimeFrequency.py->6-AudioSegments.py
Özellikleri bu şekilde ayrı scriptlerde aldım ki okunurluğu kolay olsun ve daha sonrasında yapacağım accuracy değerı yükseltme işlemlerinde ekleme çıkarmayı daha kolay şekilde yapabileyim.
En son csv dosyamı kullanarak neural network kullanarak modelimi eğittim . -> neural.py
Doğruluk Skoru: 0.8357142857142857
Sınıflandırma Raporu:
               precision    recall  f1-score   support

           0       0.79      0.91      0.85        70
           1       0.90      0.76      0.82        70

    accuracy                           0.84       140
   macro avg       0.84      0.84      0.83       140
weighted avg       0.84      0.84      0.83       140


