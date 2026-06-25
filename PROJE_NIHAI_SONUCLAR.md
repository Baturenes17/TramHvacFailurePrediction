# PROJE NİHAİ SONUÇLAR — Tram HVAC Arıza Tahmini

Bu belge projede **ne yaptığımızı, hangi araçları/teknikleri kullandığımızı** özetler ve
istenen komutların **gerçek çıktılarını** (hangi model, hangi teknikte ne sonuç alındığı) kaydeder.

> Çalıştırma tarihi: 2026-06-16
> Ayrıntılı pipeline dokümantasyonu için bkz. [PROJE.md](PROJE.md)

---

## 1. Projede Ne Yaptık? Ne Kullandık?

### Problem
- **Amaç:** Bir tramvay aracının **önümüzdeki 30 gün içinde** HVAC arızası yapıp yapmayacağını tahmin etmek.
- **Problem tipi:** İkili sınıflandırma. Hedef = `failure_next_30d`.
- **Tamamlayıcı problem:** Filo genelinde önümüzdeki dönemde **kaç arıza** beklenir? (zaman serisi tahmini).

### Veri
- **Kaynak:** `3_years_data.csv` ve türevleri (67.653 araç-gün, **80 araç**, 2023-01-01 → 2026-01).
- **Format:** `;` ayraçlı, `,` ondalık (Türkçe/Avrupa).
- **Sınıf dengesi:** `failure_next_30d` pozitif oranı ≈ **%20** → dengesiz veri.
- Bu deneylerde kullanılan veri setleri:
  - `3_years_data_no7d.csv` — `failure_next_7d` sütunu çıkarılmış tam set (67.653 satır).
  - `3_years_data_no7d_undersampled.csv` — yukarıdakinin undersample (azınlık dengesine yakın) edilmiş hali (27.186 satır).

### Feature Engineering (sızıntısız / causal)
- **27 özellik** üretildi; hepsi yalnızca geçmiş/aynı-gün bilgisinden (gelecek sızıntısı yok).
- Takvim/mevsim döngüsel kodlama (`doy_sin/cos`, `season`, `is_summer`), hava-stres
  (`temp_sq`, `is_hot/is_cold`, `temp_x_humidity`, birikimli sıcak/soğuk gün sayıları),
  kullanım trendi (`km_7d_30d_ratio`, `km_roll30_std`), arıza geçmişi
  (`veh_past_failure_rate`, `veh_failures_last_90d`), bakım yoğunluğu etkileşimleri.
- Araç-bazlı yuvarlanan istatistikler `shift(1)` ile kaydırıldı (sızıntı engellendi).

### Önişleme
- `sklearn ColumnTransformer`: sayısal → `SimpleImputer(median)`; kategorik → `SimpleImputer(most_frequent)` + `OrdinalEncoder`.

### Modeller / Kütüphaneler
- **LightGBM** (ana model), ayrıca seçenek olarak **LogReg, XGBoost, CatBoost, Ensemble (soft-voting)**.
- **imbalanced-learn**: `SMOTENC` (oversampling) + `RandomUnderSampler` (undersampling).
- **Facebook Prophet**: filo geneli zaman serisi tahmini.
- **Optuna** (hiperparametre araması), **SHAP** (özellik önemi).
- Sınıf dengesizliği: `scale_pos_weight = negatif/pozitif`.

### Teknikler
- **Takvim bazlı 80/20 train/test split** (kronolojik, sızıntısız).
- **Eşik seçim modları:** `alarm` (en riskli %k), `fixed` (train recall≥0.80), `precision` (hedef precision).
- **Sınıf dengeleme:** undersampling, SMOTE oversampling, ikisinin kombinasyonu.
- **Değerlendirme metrikleri:** ROC-AUC, PR-AUC (AP), precision/recall/F1, precision-recall ödünleşim tablosu.
- **Prophet için:** holdout MAE/RMSE/MAPE + naive & mevsimsel-naive baseline karşılaştırması + cross-validation.

---

## 2. Sonuç Özet Tablosu (Sınıflandırma)

Tüm koşular: `--model lightgbm --test-end 2025-12-01 --threshold-mode fixed` (sabit eşik, train recall≥0.80).

| # | Veri seti | Teknik | Train ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Precision (arıza) | Test Recall (arıza) | Test F1 (arıza) |
|---|---|---|---|---|---|---|---|---|
| 1 | no7d (tam) | Baz (dengeleme yok) | 0.9478 | **0.7109** | 0.3357 | 0.3280 | 0.6342 | 0.4324 |
| 2 | no7d_undersampled | Undersample | 0.9467 | 0.6869 | **0.6449** | 0.6073 | 0.6549 | 0.6302 |
| 3a | no7d (tam) | + SMOTE (1:1) | 0.9744 | 0.6998 | 0.3246 | 0.3638 | 0.1935 | 0.2526 |
| 3b | no7d_undersampled | + SMOTE (1:1) | 0.9486 | 0.6890 | 0.6406 | 0.6166 | 0.6878 | 0.6502 |

**Yorum:**
- **En yüksek Test ROC-AUC:** baz model (tam veri, dengeleme yok) → **0.7109**. Ama tam veride
  test PR-AUC ve precision/recall düşük (sınıf dengesizliği nedeniyle).
- **Undersampling**, test PR-AUC'yi (0.34 → 0.64) ve arıza-sınıfı precision/recall dengesini
  belirgin yükseltir — fakat bu, **test setinin de dengelendiği** (negatifler azaltıldığı) için
  optimist bir görüntüdür; ROC-AUC hafif düşer (0.71 → 0.69).
- **Tam veriye SMOTE** uygulamak precision'ı bir tık artırsa da **recall'ı çökertir** (0.63 → 0.19);
  PR-AUC iyileşmez. SMOTE bu veride faydalı değil.
- **Undersampled veriye SMOTE** neredeyse etkisizdir (azınlık zaten çoğunlukla eşit, +0 sentetik üretildi);
  sonuçlar undersample-only ile pratikçe aynıdır.

---

## 3. Komut Çıktıları (Tam)

### Komut 1 — `3_years_data_no7d.csv` (undersample UYGULANMAMIŞ)

```
python train_failure_model.py --data 3_years_data_no7d.csv --model lightgbm --test-end 2025-12-01 --threshold-mode fixed
```

```
Data shape: (67653, 15)
[test-end] Test 2025-12-01 sonrası dışlandı.
Train size: 51837  dates: 2023-01-01 - 2025-05-26
Test  size: 13658  dates: 2025-05-27 - 2025-12-01
Model: lightgbm
[LightGBM] sabit ağaç sayısı: 200 (early stopping kapalı)
Eşik modu — sabit eşik (train recall>=0.80)=0.599 | referans: F1-opt=0.585

--- TRAIN ---
==== Train ROC-AUC: 0.9478 | PR-AUC (AP): 0.8295 ====
[eşik=0.599 | işaretlenen oran=0.244]
              precision    recall  f1-score   support
           0     0.9440    0.9072    0.9252     40797
           1     0.7002    0.8010    0.7472     11040
    accuracy                         0.8846     51837

Precision/Recall ödünleşimi (arıza sınıfı):
  recall>=  precision   eşik
   0.90       0.616     0.520
   0.80       0.700     0.599
   0.70       0.774     0.656
   0.60       0.829     0.696
   0.50       0.869     0.728

--- TEST ---
==== Test ROC-AUC: 0.7109 | PR-AUC (AP): 0.3357 ====
[eşik=0.599 | işaretlenen oran=0.361]
              precision    recall  f1-score   support
           0     0.8929    0.7013    0.7856     11105
           1     0.3280    0.6342    0.4324      2553
    accuracy                         0.6888     13658

Precision/Recall ödünleşimi (arıza sınıfı):
  recall>=  precision   eşik
   0.90       0.225     0.360
   0.80       0.262     0.467
   0.70       0.309     0.558
   0.60       0.337     0.617
   0.50       0.366     0.658

Model training completed.
Train ROC-AUC: 0.947765068820775
Test ROC-AUC: 0.7109274730949261

[Tahmin] Alarm eşiği (en riskli %10): 0.1853, alarm sayısı: 8
En riskli 3 araç: 3876 (BOZANKAYA, 0.326), 3880 (BOZANKAYA, 0.317), 3808 (SIRIO, 0.308)
```

---

### Komut 2 — `3_years_data_no7d_undersampled.csv` (undersample uygulanmış)

```
python train_failure_model.py --data 3_years_data_no7d_undersampled.csv --model lightgbm --test-end 2025-12-01 --threshold-mode fixed
```

```
Data shape: (27186, 15)
[test-end] Test 2025-12-01 sonrası dışlandı.
Train size: 21287  dates: 2023-01-01 - 2025-05-26
Test  size: 5345  dates: 2025-05-27 - 2025-12-01
Model: lightgbm
[LightGBM] sabit ağaç sayısı: 200 (early stopping kapalı)
Eşik modu — sabit eşik (train recall>=0.80)=0.587 | referans: F1-opt=0.471

--- TRAIN ---
==== Train ROC-AUC: 0.9467 | PR-AUC (AP): 0.9441 ====
[eşik=0.587 | işaretlenen oran=0.459]
              precision    recall  f1-score   support
           0     0.8086    0.9092    0.8560     10247
           1     0.9048    0.8003    0.8493     11040
    accuracy                         0.8527     21287

Precision/Recall ödünleşimi (arıza sınıfı):
  recall>=  precision   eşik
   0.90       0.859     0.505
   0.80       0.905     0.587
   0.70       0.934     0.644
   0.60       0.956     0.689
   0.50       0.968     0.728

--- TEST ---
==== Test ROC-AUC: 0.6869 | PR-AUC (AP): 0.6449 ====
[eşik=0.587 | işaretlenen oran=0.515]
              precision    recall  f1-score   support
           0     0.6601    0.6128    0.6356      2792
           1     0.6073    0.6549    0.6302      2553
    accuracy                         0.6329      5345

Precision/Recall ödünleşimi (arıza sınıfı):
  recall>=  precision   eşik
   0.90       0.529     0.406
   0.80       0.541     0.475
   0.70       0.581     0.553
   0.60       0.640     0.626
   0.50       0.697     0.684

Model training completed.
Train ROC-AUC: 0.9466640996375041
Test ROC-AUC: 0.6869023548900839

[Tahmin] Alarm eşiği (en riskli %10): 0.2274, alarm sayısı: 8
En riskli 3 araç: 3878 (BOZANKAYA, 0.386), 3881 (BOZANKAYA, 0.314), 3879 (BOZANKAYA, 0.291)
```

---

### Komut 3a — `3_years_data_no7d.csv` + SMOTE

```
python train_failure_model.py --data 3_years_data_no7d.csv --model lightgbm --test-end 2025-12-01 --threshold-mode fixed --smote
```

```
[SMOTE] train 51837 -> 81594 satır | arıza(1): 11040 -> 40797 (+29757 sentetik) | 1-oranı: 21.3% -> 50.0%
[LightGBM] sabit ağaç sayısı: 200 (early stopping kapalı)
Eşik modu — sabit eşik (train recall>=0.80)=0.563 | referans: F1-opt=0.398

--- TRAIN ---
==== Train ROC-AUC: 0.9744 | PR-AUC (AP): 0.9778 ====
[eşik=0.563 | işaretlenen oran=0.409]
              precision    recall  f1-score   support
           0     0.8333    0.9855    0.9030     40797
           1     0.9822    0.8029    0.8835     40797
    accuracy                         0.8942     81594

--- TEST ---
==== Test ROC-AUC: 0.6998 | PR-AUC (AP): 0.3246 ====
[eşik=0.563 | işaretlenen oran=0.099]
              precision    recall  f1-score   support
           0     0.8326    0.9222    0.8751     11105
           1     0.3638    0.1935    0.2526      2553
    accuracy                         0.7860     13658

Precision/Recall ödünleşimi (arıza sınıfı):
  recall>=  precision   eşik
   0.90       0.226     0.246
   0.80       0.252     0.293
   0.70       0.304     0.359
   0.60       0.330     0.407
   0.50       0.335     0.443

Model training completed.
Train ROC-AUC: 0.9743855823608057
Test ROC-AUC: 0.6997806607970459
```
> **Not:** SMOTE train PR-AUC'yi şişiriyor (0.98) ama test'te recall çöküyor (0.19). Sentetik
> azınlık örnekleri gerçek test dağılımını yansıtmıyor → SMOTE bu veride faydasız.

---

### Komut 3b — `3_years_data_no7d_undersampled.csv` + SMOTE

```
python train_failure_model.py --data 3_years_data_no7d_undersampled.csv --model lightgbm --test-end 2025-12-01 --threshold-mode fixed --smote
```

```
[SMOTE] train 21287 -> 22080 satır | arıza(1): 11040 -> 11040 (+0 sentetik) | normal(0): 10247 (sabit) | 1-oranı: 51.9% -> 50.0%
[LightGBM] sabit ağaç sayısı: 200 (early stopping kapalı)
Eşik modu — sabit eşik (train recall>=0.80)=0.589 | referans: F1-opt=0.488

--- TRAIN ---
==== Train ROC-AUC: 0.9486 | PR-AUC (AP): 0.9430 ====
[eşik=0.589 | işaretlenen oran=0.446]
              precision    recall  f1-score   support
           0     0.8215    0.9098    0.8634     11040
           1     0.8989    0.8023    0.8478     11040
    accuracy                         0.8560     22080

--- TEST ---
==== Test ROC-AUC: 0.6890 | PR-AUC (AP): 0.6406 ====
[eşik=0.589 | işaretlenen oran=0.533]
              precision    recall  f1-score   support
           0     0.6808    0.6089    0.6428      2792
           1     0.6166    0.6878    0.6502      2553
    accuracy                         0.6466      5345

Precision/Recall ödünleşimi (arıza sınıfı):
  recall>=  precision   eşik
   0.90       0.529     0.398
   0.80       0.545     0.479
   0.70       0.608     0.577
   0.60       0.648     0.642
   0.50       0.681     0.685

Model training completed.
Train ROC-AUC: 0.9485772503806973
Test ROC-AUC: 0.6889862564071484
```
> **Not:** Undersample edilmiş veri zaten ~1:1 dengede olduğu için SMOTE neredeyse hiç sentetik
> örnek üretmedi (+0). Sonuç undersample-only (Komut 2) ile pratikçe aynı.

---

## 4. Prophet — Filo Geneli Arıza Tahmini

> **Not:** Prophet komutları, PROJE.md'deki bulguya göre en temiz sonucu veren
> `3_years_data_no2025.csv` (2025 komple çıkarılmış, 2023–2024) seti ile çalıştırılmıştır.

### Komut 4 — Prophet (varsayılan, haftalık)

```
python forecast_failures_prophet.py --data 3_years_data_no2025.csv
```

```
======================================================================
FİLO GENELİ ARIZA TAHMİNİ — Prophet
======================================================================
Frekans               : W  (D=günlük, W=haftalık)
Toplam arıza olayı    : 385
Seri uzunluğu         : 106 periyod (2023-01-01 - 2025-01-05)
Periyod başına y (ort): 3.63  (min=0, max=24)

Train  : 84 periyod (2023-01-01 - 2024-08-04)
Holdout: 22 periyod (2024-08-11 - 2025-01-05)

----------------------------------------------------------------------
HOLDOUT DEĞERLENDİRME
----------------------------------------------------------------------
Model                        MAE      RMSE     MAPE%
Prophet                    4.375     7.144    122.48
Naive (ortalama)           3.864     7.151     74.41
Mevsimsel naive            5.227     8.337    106.00
-> En düşük MAE: Naive (ortalama) (3.864).
   Mevsimsel naive bile düz ortalamayı geçemedi — bu holdout penceresinde
   yıllar arası patern kaymış olabilir.

----------------------------------------------------------------------
GELECEK TAHMİNİ (sonraki periyodlar)
----------------------------------------------------------------------
        ds     yhat  yhat_lower  yhat_upper
2025-01-12 5.143410    0.005736    9.933508
2025-01-19 5.436862    0.594767   10.251577
2025-01-26 4.728675    0.052760    9.473614
2025-02-02 4.243836    0.000000    9.198890
2025-02-09 4.585557    0.000000    9.524327
2025-02-16 5.066344    0.454161    9.880536
2025-02-23 4.738935    0.145522    9.532889
2025-03-02 3.725808    0.000000    8.470124
```

> **Yorum:** Holdout MAE'de Prophet (4.375) hâlâ düz ortalama baseline'ı (3.864) **geçemiyor**,
> ama 2025'li sete göre çok daha yakın (5.74 → 4.375). Tek-yıl holdout'u yanıltıcı; asıl güvenilir
> ölçüm aşağıdaki çok-katlı CV'dir.

### Komut 5 — Prophet + Cross-Validation (`--cv`)

```
python forecast_failures_prophet.py --data 3_years_data_no2025.csv --cv
```

Holdout değerlendirme yukarıdakiyle aynı (Prophet 4.375 / Naive 3.864 / Mevsimsel naive 5.227).
Cross-validation (genişleyen pencere, initial=367 gün, period=73 gün, horizon=56 gün, 5 kat):

```
----------------------------------------------------------------------
CROSS-VALIDATION (initial=367 days, period=73 days, horizon=56 days)
----------------------------------------------------------------------
horizon      mae      rmse  coverage
 6 days 3.257174  4.489583      0.50
 9 days 1.780116  1.959376      0.75
12 days 1.344544  1.513977      1.00
13 days 1.237733  1.340880      1.00
14 days 1.453212  1.516255      1.00
19 days 1.314810  1.418963      1.00
21 days 6.197709 10.577114      0.75
24 days 6.253766 10.588732      0.75
27 days 2.268573  3.386055      0.75
33 days 1.188947  1.458245      0.75
35 days 4.564016  7.551291      0.75
40 days 1.812075  2.393818      0.75
45 days 6.384092 10.134813      0.50
48 days 7.159553 10.393892      0.25
49 days 7.193947 10.395146      0.25
54 days 1.500127  2.146787      0.75
56 days 4.240588  5.035887      0.50
(... tüm horizon değerleri çıktıda mevcut ...)

CV ortalama MAE: 3.361 (çok-katlı; tek-yıl holdout'tan daha güvenilir)
```

> **Yorum:** Çok-katlı CV ortalama MAE = **3.361** — 2025'li sete göre belirgin daha iyi (4.910 → 3.361),
> PROJE.md'deki "en temiz sonuç 2025'siz sette" bulgusunu doğruluyor. Kısa-orta ufuklarda hata düşük
> (12–19 gün: MAE ~1.2–1.5) ama bazı ufuklarda (21–24, 45–49 gün) hata yükseliyor (MAE 6–7) — agregat
> seride yıllar-arası non-stasyonarite hâlâ etkili. Yine de 2025 verisi çıkarıldığında 2025 sonu
> veri çöküşü kaynaklı bozulma ortadan kalkıyor ve tahmin kararlılaşıyor.

---

## 5. Genel Sonuç

- **Sınıflandırma (araç-bazlı):** LightGBM ana model. Tam veride **Test ROC-AUC ≈ 0.71** elde edilir;
  bu veriden çıkabilecek tavana yakındır. Undersampling, dengeli bir test üzerinde daha yüksek
  precision/recall (F1 ≈ 0.63) gösterir ama bu test setinin de dengelenmesinden gelen optimist bir
  görüntüdür. **SMOTE bu veride fayda sağlamadı.**
- **Darboğaz model değil, veri/sinyaldir:** Yeni özellikler ve sınıf dengeleme teknikleri precision+recall'ı
  birlikte yükseltmedi. Gerçek iyileşme için **yeni sinyal** (HVAC telemetri/hata kodları) gerekir.
- **Zaman serisi (filo geneli):** Prophet mevsimselliği öğreniyor ama küçük/gürültülü haftalık sayımlar
  ve 2025 sonu veri çöküşü nedeniyle naive baseline'ı geçemiyor.
- **Operasyonel öneri:** Sabit eşik yerine **alarm-oranı (en riskli %10) + düzenli yeniden eğitim**;
  zamansal non-stasyonarite nedeniyle.
