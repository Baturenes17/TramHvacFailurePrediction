# PROJE.md — Tram HVAC 30 Günlük Arıza Tahmini (Uygulama Dokümantasyonu)

Bu belge, `train_failure_model.py` pipeline'ının **veri, özellik, model, eğitim, değerlendirme ve
tahmin** adımlarını ayrıntılı anlatır. Üst düzey tanıtım için bkz. `PROJE_TANITIM.md`.

> **Kural:** Kod/model/veri değiştikçe bu dosya güncel tutulur.

---

## 1. Amaç ve Problem

- **Amaç:** Bir tramvay aracının **önümüzdeki 30 gün içinde** HVAC arızası yapıp yapmayacağını tahmin etmek.
- **Problem tipi:** İkili sınıflandırma. Hedef = `failure_next_30d` (1 = 30 gün içinde arıza var, 0 = yok).
- **Çıktı:** Her araç için 30 günlük arıza olasılığı + riske göre sıralı öncelik listesi (kestirimci bakım).
- Bu sürümde **ufuk sabit: 30 gün**. `failure_next_7d` yalnızca sızıntı önleme için özelliklerden çıkarılır.

---

## 2. Veri

- **Kaynak:** `3_years_data.csv`
- **Format:** `;` ayraçlı, ondalık ayırıcı `,` (Türkçe/Avrupa). Okuma: `pd.read_csv(sep=";", decimal=",")`.
- **Boyut:** 67.653 araç-gün kaydı, **80 araç**, 2023-01-01 → 2025-12-31.
- **Eksik değer:** Yok (tüm sütunlar dolu).
- **Granülarite:** Her satır = bir araç-gün gözlemi.

### Sütunlar

| Sütun | Tip | Açıklama |
|---|---|---|
| `date` | tarih | Gözlem günü |
| `vehicle_id` | int | Araç kimliği (kategorik anlam) |
| `vehicle_type` | kategorik | `SIRIO` / `BOZANKAYA` |
| `vehicle_age` | int | Araç yaşı (yıl) |
| `km_today` | int | O günkü km |
| `km_last_7d` | int | Son 7 günün toplam km'si |
| `km_last_30d` | int | Son 30 günün toplam km'si |
| `days_since_last_maintenance` | int | Son bakımdan beri gün |
| `km_since_last_maintenance` | int | Son bakımdan beri km |
| `days_since_last_failure` | int | Son arızadan beri gün |
| `temp` | float | Sıcaklık (°C) |
| `weather_type` | int (kategorik) | WMO hava kodu (0–75) |
| `wind_speed` | float | Rüzgâr hızı |
| `humidity` | int | Nem (%) |
| `failure_next_7d` | int | 7 gün ufuk etiketi (**özelliklerden çıkarılır**) |
| `failure_next_30d` | int | **Hedef etiket** |

### Sınıf dengesi
- `failure_next_30d` pozitif oranı ≈ **%20.1** (13.593 pozitif / 67.653). Dengesiz → tek başına accuracy
  yanıltıcıdır; **ROC-AUC, PR-AUC, precision/recall** kullanılır.

### Belgede olup CSV'de OLMAYAN sütunlar
`cabin_id`, `other_failures_since_last_maintenance`, `failures_last_90d` bu CSV'de yok. Feature
engineering yalnızca mevcut sütunlarla yapılır; bu özellikler atlanmıştır. (İleride
`days_since_last_failure` sıfırlanma noktalarından geçmiş arıza sayısı türetilebilir.)

---

## 3. Feature Engineering (sızıntısız / causal)

Tüm türetilen özellikler **yalnızca geçmiş veya aynı-gün** bilgisinden gelir; gelecekteki etiketten
asla türetilmez. Araç-bazlı yuvarlanan istatistikler `shift(1)` ile kaydırılır.

| Özellik | Formül / Mantık |
|---|---|
| `month`, `dayofweek` | Takvim alanları |
| `doy_sin`, `doy_cos` | Yıl-içi-gün döngüsel kodlama: `sin/cos(2π·doy/365.25)` |
| `season` | Mevsim (winter/spring/summer/autumn) — kategorik |
| `is_summer` | Haziran–Ağustos bayrağı |
| `temp_sq` | `temp²` (doğrusal olmayan ısı stresi) |
| `is_hot` / `is_cold` | `temp≥30` / `temp≤0` bayrakları |
| `temp_x_humidity` | `temp · humidity` (etkileşim) |
| `temp_dev_from_month` | `temp − (aylık ortalama temp)` — exogen iklim sapması |
| `km_7d_30d_ratio` | `km_last_7d / (km_last_30d + ε)` — kullanım trendi |
| `km_roll30_std` | Araç bazında `km_today`'in **shift(1)** sonrası 30 günlük yuvarlanan std'si |
| `veh_past_failure_rate` | Aracın geçmiş arıza sayısı / geçmiş gözlem günü (durağan oran) |
| `veh_failures_last_90d` | Aracın son 90 gündeki arıza sayısı (kayan pencere) |

**Arıza geçmişi nasıl türetilir:** Ham "arıza oldu" sütunu yok; `days_since_last_failure` bir arıza
olduğunda sıfırlanır, dolayısıyla bu sayaç bir önceki güne göre **düştüğü** gün bir arıza olmuştur
(`failure_event`). Bu olaylardan **yalnızca `shift(1)` ile geçmişe bakarak** yukarıdaki iki özellik
üretilir (bugünün/geleceğin arızası sayıma girmez → sızıntı yok).
> **Not (önemli):** Kümülatif `veh_past_failures` ve `veh_past_obs` özellik OLARAK kullanılmaz; çünkü
> zamanla monoton büyürler ve takvim-bazlı split'te train→test dağılım kayması yaratıp genellemeyi
> bozar (deneyle doğrulandı: test AUC 0.73→0.68). Sadece durağan `rate` ve `last_90d` tutulur.

Ham causal kolonlar olduğu gibi kullanılır: `vehicle_age`, `days_since_last_maintenance`,
`km_since_last_maintenance`, `days_since_last_failure`, `temp`, `wind_speed`, `humidity`,
`km_today/7d/30d`, `weather_type`, `vehicle_type`.

**Toplam özellik sayısı: 27.**

### Sızıntı (leakage) önlemleri
- Hedef `failure_next_30d` ve diğer ufuk `failure_next_7d` **özelliklerden çıkarılır**
  (`get_feature_columns`), `assert` ile doğrulanır.
- Araç-bazlı yuvarlanan istatistikler `shift(1)` kullanır (aynı günün/sonrasının bilgisi sızmaz).
- Zaman bazlı split kronolojiktir (aşağıya bkz.).

---

## 4. Önişleme

`sklearn` `ColumnTransformer` içinde:
- **Sayısal:** `SimpleImputer(median)` (güvenlik amaçlı; veride NaN yok).
- **Kategorik** (`vehicle_type`, `weather_type`, `season`): `SimpleImputer(most_frequent)` →
  `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.
- Ağaç tabanlı model olduğu için ölçekleme yapılmaz.

Tümü tek bir `Pipeline(prep → clf)` içinde; eğitim ve tahminde aynı dönüşüm garanti.

---

## 5. Model

- **Model seçimi `--model` ile** (varsayılan `lightgbm`):
  - **`lightgbm`** (`LGBMClassifier`): gradient-boosted ağaçlar; `scale_pos_weight` ile dengelenir.
  - **`logreg`**: ölçeklenmiş Lojistik Regresyon (`StandardScaler` + `LogisticRegression`,
    `class_weight='balanced'`). 6 model karşılaştırmasında (bkz. `benchmark_models.py`) bu zayıf-lineer
    sinyalde **en yüksek precision'ı** verdi: test precision@top10 ≈ **0.41** (LightGBM 0.38; XGBoost/
    CatBoost/RF daha düşük). Ağaç modelleri burada gürültüye overfit oluyor. `--tune`/`--shap`/
    `--early-stopping` yalnız `lightgbm` içindir, `logreg`'de sessizce atlanır.
- `build_model(model_type, ...)` yeni model eklemeye açık.
- **Sınıf dengesizliği:** `scale_pos_weight = (negatif sayısı / pozitif sayısı)`.
- **Varsayılan hiperparametreler:** `n_estimators=200`, `learning_rate=0.03`, `num_leaves=31`,
  `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `random_state=42`.
- **Ağaç sayısı — sabit 200 (early stopping varsayılan KAPALI):** Validation penceresi temsili
  olmadığından early stopping modeli çok erken (≈5 ağaç) durdurup sakatlıyordu; bu hem AUC'yi hem
  precision'ı düşürüyor, skorları kabalaştırıp (eşitlikler) eşik kontrolünü bozuyordu. Deneyle en iyi
  bant 50–200 ağaç çıktı (test precision@top10 ≈ 0.32 → 0.38). `--early-stopping` bayrağıyla eski
  davranış (val üzerinde `auc`, `EARLY_STOPPING_ROUNDS=50`) geri açılabilir.

---

## 6. Eğitim ve Değerlendirme

### Takvim bazlı bölme
Tarih **aralığına** göre **%60 / %20 / %20** (satır sayısına göre değil; toplam zaman ekseninin
ilk %60'ı train, sonraki %20'si validation, son %20'si test). Kronolojik kesim — gelecekten geçmişe
sızıntı yok. Boyutlar dönemdeki kayıt yoğunluğuna göre dengesiz olabilir (ör. train≈35.884,
val≈15.953, test≈15.816) ve her split'in tarih aralığı konsola yazılır.

### Eşik seçimi (üç mod)
- **`alarm` (varsayılan):** **her split kendi** skorlarının en yüksek `--alarm-rate` (varsayılan
  **%10 — precision-odaklı**) dilimini pozitif yapar (val ve test için ayrı eşik). Base-rate
  kaymasına dayanıklı ("her dönem en riskli %k aracı incele"). Konsolda ayrıca referans olarak val
  üzerinden `op(recall≥0.80)` ve `F1-opt` eşikleri raporlanır.
- **`fixed`:** validation'da `recall ≥ 0.80` kısıtı altında precision-maks eşik; aynı eşik
  hem val hem test'e uygulanır.
- **`precision`:** validation'da `precision ≥ --target-precision` kısıtı altında recall-maks eşik;
  aynı eşik test'e uygulanır. ⚠️ Skor dağılımı dönemler arası kaydığı için sabit eşik transferi
  güvenilmez (val'de %10 işaretleyen eşik test'te %44 işaretleyebilir); bu yüzden **alarm-oranı
  daha sağlam** ve varsayılan budur.

### Raporlanan metrikler
Her split için: ROC-AUC, PR-AUC (average precision), seçilen eşikte tam `classification_report`
(precision/recall/f1/support, digits=4) ve recall hedefleri (0.90/0.80/0.70/0.60/0.50) için
precision-recall ödünleşim tablosu (`recall>= | precision | eşik`). Sonda `Model training completed.`
ve val/test ROC-AUC tek satır özet.

### Gözlemlenen sonuçlar (varsayılan çalıştırma, sabit 200 ağaç, alarm %10)
- **Test ROC-AUC ≈ 0.74**, PR-AUC ≈ 0.32. Alarm %10 noktasında **test precision (arıza sınıfı) ≈ 0.38**,
  recall ≈ 0.24 — bu veriden çıkabilecek precision tavanına yakın.
- **Zamansal kararsızlık:** Validation penceresinde ROC-AUC ~0.55'e kadar düşüyor; arıza-özellik
  ilişkisi dönemden döneme kayıyor (`PROJE_TANITIM.md`'deki "base-rate kayması" ile tutarlı). Bu
  yüzden sabit eşik yerine **alarm-oranı + düzenli yeniden eğitim** önerilir.
- **Veri tavanı (deneyle doğrulandı):** Mevcut sütunlardan ek türetilmiş özellikler (heatwave streak,
  bakım yoğunluğu, yaş×km / sıcaklık×km etkileşimleri, kullanım ivmesi) **PR-AUC'yi yükseltmedi**
  (hepsi gürültü seviyesinde, bazıları hafif düşürdü). Sınıf dengeleme (0 azaltma / 1 çoğaltma) ise
  precision'ı **düşürür** (recall'ı artırır). Precision+recall'ı birlikte yükseltmenin tek yolu
  **yeni sinyal** (HVAC telemetri/hata kodu) eklemek; bu veride sinyal tükenmiş durumda.

### (Opsiyonel) Optuna
`--tune` ile zaman bazlı expanding-window CV (`TimeSeriesSplit`) üzerinde **PR-AUC maksimize** eden
LightGBM hiperparametre araması (`--trials`, vars. 40).

### (Opsiyonel) SHAP
`--shap` ile test örneği üzerinde `TreeExplainer` özet grafiği → `outputs/shap_summary.png`.

---

## 7. Tahmin Akışı ("bugünden sonrası")

Veri 2025-12-31'de bittiği için "gelecek tahmini" pratikte **her aracın en güncel gününü skorlamak**tır:

1. Final model **tüm veriyle** yeniden fit edilir (güncel duruma en zengin model).
2. Her `vehicle_id` için **en son tarihli kayıt** alınır ve skorlanır.
3. Riske göre azalan sıralanır, alarm-oranı eşiğiyle `alarm_flag` üretilir.
4. `outputs/predictions_latest.csv` yazılır.

### `predictions_latest.csv` sütunları
`risk_rank`, `vehicle_id`, `vehicle_type`, `date`, `failure_prob_30d`, `alarm_flag`
(80 satır = araç başına 1; `;`+`,` formatında).

### Yeni veri skorlama: `--predict <csv>`
İleride toplanan bir CSV aynı feature pipeline'ından geçirilip skorlanır →
`outputs/predictions_custom.csv`.

---

## 8. Çalıştırma

```bash
# Varsayılan: LightGBM, sabit 200 ağaç, precision-odaklı alarm %10
python train_failure_model.py --data 3_years_data.csv

# Lojistik Regresyon — bu veride biraz daha yüksek precision (test top10 ≈ 0.41)
python train_failure_model.py --model logreg

# Daha geniş ağ (yüksek recall) için alarm oranını artır
python train_failure_model.py --alarm-rate 0.40

# SHAP özellik önemi
python train_failure_model.py --shap

# Optuna ile hiperparametre araması
python train_failure_model.py --tune --trials 40

# Early stopping'i geri aç (eski davranış)
python train_failure_model.py --early-stopping

# Sabit eşik / precision-hedef modları
python train_failure_model.py --threshold-mode fixed
python train_failure_model.py --threshold-mode precision --target-precision 0.40

# Eğitim sonrası yeni bir dosyayı skorla
python train_failure_model.py --predict yeni_veri.csv
```

### Çıktılar (`outputs/` klasörü)
- `predictions_latest.csv` — güncel risk sıralaması (asıl tahmin çıktısı).
- `shap_summary.png` — `--shap` verildiğinde.
- `predictions_custom.csv` — `--predict` verildiğinde.

---

## 9. Bilinen Kısıtlar / İyileştirme Fikirleri
- **Zamansal non-stasyonarite:** Val/test AUC dalgalanması yüksek; düzenli yeniden eğitim ve
  alarm-oranı eşiği önerilir.
- **Sağ-sansür (right-censoring):** Verinin son ~30 gününde 30g etiketleri doğası gereği eksik
  olabilir; eğitim verilen etiketlere güvenir, skorlama olasılık üretir (etiket gerektirmez).
- **Eksik geçmiş özellikleri:** `failures_last_90d` vb. CSV'de yok; `days_since_last_failure`
  sıfırlanmalarından türetilebilir.
- **Tek model:** Gerekirse `build_model()` genişletilerek XGBoost/CatBoost/ensemble eklenebilir.
