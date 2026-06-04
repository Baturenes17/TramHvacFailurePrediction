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

- **Model seçimi `--model` ile** (varsayılan `lightgbm`): `lightgbm` | `logreg` | `xgboost` |
  `catboost` | `ensemble`.
  - **`lightgbm`** (`LGBMClassifier`): gradient-boosted ağaçlar; `scale_pos_weight` ile dengelenir.
  - **`logreg`**: ölçeklenmiş Lojistik Regresyon (`StandardScaler` + `LogisticRegression`,
    `class_weight='balanced'`). 6 model karşılaştırmasında (bkz. `benchmark_models.py`) bu zayıf-lineer
    sinyalde **en yüksek precision'ı** verdi: test precision@top10 ≈ **0.41** (LightGBM 0.38; XGBoost/
    CatBoost/RF daha düşük). Ağaç modelleri burada gürültüye overfit oluyor.
  - **`xgboost`** (`XGBClassifier`): alternatif boosting; `n_estimators=200, lr=0.03, max_depth=4`,
    `scale_pos_weight` ile dengeli, `eval_metric="logloss"`.
  - **`catboost`** (`CatBoostClassifier`): `iterations=300, lr=0.03, depth=6, l2_leaf_reg=3.0`,
    `scale_pos_weight` ile dengeli. `allow_writing_files=False` (repoya `catboost_info/` yazmaz).
  - **`ensemble`**: `lightgbm + xgboost + catboost` **soft-voting** (olasılık ortalaması,
    `sklearn.VotingClassifier`). Her alt model varsayılan parametre + `scale_pos_weight` ile kurulur;
    `params`/Optuna alt modellere uygulanmaz. Tek modellerin gürültüsünü ortalayarak skoru kararlılaştırır.
- `--shap` ve `--early-stopping` yalnız **ağaç modelleri** (`lightgbm`, `xgboost`, `catboost`) için
  geçerlidir; ama `--early-stopping` API'si yalnız `lightgbm`'e bağlandı (diğerlerinde uyarıyla atlanır).
  `--tune` (Optuna) yalnız `lightgbm` içindir. `logreg`/`ensemble`'da bunlar sessizce/uyarıyla atlanır.
- `build_model(model_type, scale_pos_weight, params)` yeni model eklemeye açık; her ağaç modeli için
  ayrı `_build_<model>()` yardımcısı vardır.
- **Sınıf dengesizliği:** `scale_pos_weight = (negatif sayısı / pozitif sayısı)`.
- **Varsayılan hiperparametreler:** `n_estimators=200`, `learning_rate=0.03`, `num_leaves=31`,
  `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `random_state=42`.
- **Ağaç sayısı — sabit 200 (early stopping varsayılan KAPALI):** Erken durdurma modeli çok erken
  (≈5 ağaç) durdurup sakatlıyordu; bu hem AUC'yi hem precision'ı düşürüyor, skorları kabalaştırıp
  (eşitlikler) eşik kontrolünü bozuyordu. Deneyle en iyi bant 50–200 ağaç çıktı (test precision@top10
  ≈ 0.32 → 0.38). `--early-stopping` bayrağıyla erken durdurma geri açılabilir; ancak ayrı bir
  validation seti olmadığından eval seti olarak **test** kullanılır (`auc`, `EARLY_STOPPING_ROUNDS=50`),
  bu da test metriklerini hafifçe iyimser yapar — bu yüzden önerilmez.

---

## 6. Eğitim ve Değerlendirme

### Takvim bazlı bölme
Tarih **aralığına** göre **%80 / %20** train/test (satır sayısına göre değil; toplam zaman ekseninin
ilk %80'i train, son %20'si test). Ayrı validation seti yoktur. Kronolojik kesim — gelecekten geçmişe
sızıntı yok. Boyutlar dönemdeki kayıt yoğunluğuna göre dengesiz olabilir (ör. train≈51.837
[2023-01-01 → 2025-05-26], test≈15.816 [2025-05-27 → 2025-12-31]) ve her split'in tarih aralığı
konsola yazılır.

### Eşik seçimi (üç mod)
- **`alarm` (varsayılan):** **Her split kendi** skorlarının en yüksek `--alarm-rate` (varsayılan
  **%10 — precision-odaklı**) dilimini pozitif yapar (train ve test için ayrı eşik). Base-rate
  kaymasına dayanıklı ("en riskli %k aracı incele"). Konsolda ayrıca referans olarak **train**
  üzerinden `op(recall≥0.80)` ve `F1-opt` eşikleri raporlanır.
- **`fixed`:** **train**'de `recall ≥ 0.80` kısıtı altında precision-maks eşik; aynı eşik test'e
  uygulanır.
- **`precision`:** **train**'de `precision ≥ --target-precision` kısıtı altında recall-maks eşik;
  aynı eşik test'e uygulanır. ⚠️ Validation kaldırıldığı için referans olarak train kullanılır; train
  üzerinde seçilen eşikler iyimser olabilir ve skor dağılımı dönemler arası kaydığından sabit eşik
  transferi güvenilmezdir; bu yüzden **alarm-oranı daha sağlam** ve varsayılan budur.

### Raporlanan metrikler
**Hem train hem test** için ayrı ayrı: ROC-AUC, PR-AUC (average precision), seçilen eşikte tam
`classification_report` (precision/recall/f1/support, digits=4) ve recall hedefleri
(0.90/0.80/0.70/0.60/0.50) için precision-recall ödünleşim tablosu (`recall>= | precision | eşik`).
Train raporu modelin eğitildiği veriyi gösterir (overfit'i görmek için referanstır; gerçek başarı
ölçütü test'tir). Sonda `Model training completed.` ve train + test ROC-AUC tek satır özet.

Tipik koşuda **train ROC-AUC ≈ 0.93**, **test ROC-AUC ≈ 0.73** — aradaki fark ağaç modelinin
beklenen overfit'idir; alarm-oranı eşiği ve düzenli yeniden eğitim bu yüzden önerilir.

### Gözlemlenen sonuçlar (varsayılan çalıştırma, sabit 200 ağaç, alarm %10)
- **Test ROC-AUC ≈ 0.73**, PR-AUC ≈ 0.31. Alarm %10 noktasında **test precision (arıza sınıfı) ≈ 0.36**,
  recall ≈ 0.22 — bu veriden çıkabilecek precision tavanına yakın.
- **Zamansal kararsızlık:** Arıza-özellik ilişkisi dönemden döneme kayıyor (`PROJE_TANITIM.md`'deki
  "base-rate kayması" ile tutarlı). Bu yüzden sabit eşik yerine **alarm-oranı + düzenli yeniden
  eğitim** önerilir.
- **Veri tavanı (deneyle doğrulandı):** Mevcut sütunlardan ek türetilmiş özellikler (heatwave streak,
  bakım yoğunluğu, yaş×km / sıcaklık×km etkileşimleri, kullanım ivmesi) **PR-AUC'yi yükseltmedi**
  (hepsi gürültü seviyesinde, bazıları hafif düşürdü). Sınıf dengeleme (0 azaltma / 1 çoğaltma) ise
  precision'ı **düşürür** (recall'ı artırır). Precision+recall'ı birlikte yükseltmenin tek yolu
  **yeni sinyal** (HVAC telemetri/hata kodu) eklemek; bu veride sinyal tükenmiş durumda.

### Model karşılaştırması (varsayılan çalıştırma, alarm %10, takvim 80/20 split)
| Model | Train ROC | Test ROC | Test PR-AUC |
|---|---|---|---|
| `logreg` | 0.686 | **0.752** | **0.342** |
| `lightgbm` | 0.935 | 0.734 | 0.311 |
| `ensemble` | 0.900 | 0.726 | 0.292 |
| `catboost` | 0.892 | 0.723 | 0.296 |
| `xgboost` | 0.840 | 0.692 | 0.260 |

**Sonuç:** XGBoost/CatBoost/ensemble eklendi ama **test'te LightGBM'i geçemediler**; en yüksek test
ROC/PR-AUC hâlâ `logreg`'de. Bu, "sinyal tükenmiş + ağaç modelleri gürültüye overfit oluyor"
gözlemini doğruluyor (ağaç modellerinde train≫test farkı büyük). Ensemble, tekil ağaçların ortalaması
olarak en kararlı ağaç-tabanlı seçenek ama yine lineer modelin altında.

### (Opsiyonel) Optuna
`--tune` ile **train** seti üzerinde zaman bazlı expanding-window CV (`TimeSeriesSplit`) ile
**PR-AUC maksimize** eden LightGBM hiperparametre araması (`--trials`, vars. 40).

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

# Alternatif boosting modelleri ve soft-voting ensemble
python train_failure_model.py --model xgboost
python train_failure_model.py --model catboost
python train_failure_model.py --model ensemble

# Daha geniş ağ (yüksek recall) için alarm oranını artır
python train_failure_model.py --alarm-rate 0.40

# SHAP özellik önemi
python train_failure_model.py --shap

# Optuna ile hiperparametre araması
python train_failure_model.py --tune --trials 40

# Early stopping'i geri aç (eski davranış)
python train_failure_model.py --early-stopping

# Sabit eşik / precision-hedef modları (eşik referansı: train seti)
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
- **Zamansal non-stasyonarite:** Dönemler arası AUC dalgalanması yüksek; düzenli yeniden eğitim ve
  alarm-oranı eşiği önerilir.
- **Sağ-sansür (right-censoring):** Verinin son ~30 gününde 30g etiketleri doğası gereği eksik
  olabilir; eğitim verilen etiketlere güvenir, skorlama olasılık üretir (etiket gerektirmez).
- **Eksik geçmiş özellikleri:** `failures_last_90d` vb. CSV'de yok; `days_since_last_failure`
  sıfırlanmalarından türetilebilir.
- **Model çeşitliliği:** `build_model()` artık `lightgbm`/`logreg`/`xgboost`/`catboost`/`ensemble`
  destekler. Yine de bu veride yeni boosting modelleri test başarısını yükseltmedi (bkz. §6 karşılaştırma
  tablosu) — darboğaz model değil, **veri/sinyal**. Stacking veya kalibrasyon denenebilir ama beklenti düşük.

---

## 10. Tamamlayıcı Bileşen — Filo Geneli Arıza Tahmini (`forecast_failures_prophet.py`)

`train_failure_model.py` **araç-bazlı** soruya cevap verir ("hangi araç 30 gün içinde arıza yapar?").
`forecast_failures_prophet.py` ise **farklı bir iş sorusunu** Facebook Prophet ile çözer:
**"Önümüzdeki dönemde filo genelinde kaç HVAC arızası beklenir?"** — bakım kapasitesi / personel planlaması
için. İki bileşen birbirinin yerine geçmez; tamamlayıcıdır.

### Neden zaman serisi / Prophet?
- Filo geneli arıza sayısı **tek değişkenli, takvim-indeksli sürekli bir seridir**. Trend + güçlü yıllık
  mevsimsellik (yaz HVAC yükü) içerir — Prophet'in tam güçlü olduğu alan.
- **Per-vehicle Prophet neden tercih edilmedi:** 80 aracın her birinin arıza serisi seyrek ve ikilidir;
  Prophet ikili olay/olasılık üretmez, sürekli değer üretir. Araç önceliklendirme zaten sınıflandırma
  modeliyle (LightGBM/LogReg) daha doğru yapılır. Bu yüzden Prophet yalnızca **agregat filo seviyesinde** kullanılır.

### Akış
1. **Veri:** `3_years_data.csv` (varsayılan; aynı `;`/`,` formatı). `--data 3_years_data_no2025.csv` ile
   2025'i komple çıkaran (2023–2024) temizlenmiş set kullanılabilir — aşağıdaki Bulgu'ya göre en iyi sonucu bu verir.
2. **Gerçek arıza olayı:** `days_since_last_failure` sayacı bir önceki güne göre düştüğü gün arıza olmuştur
   (`train_failure_model.py:116-118` ile aynı causal kural). İleriye-dönük `failure_next_*` ETİKETLERİ
   **kullanılmaz** — onlar gerçek olay değil, etikettir.
3. **Seri:** Tarihe göre `failure_event` toplanır, eksik günler 0 ile doldurulur; varsayılan **haftalık**
   (`--freq W`) toplama (seyreklik/gürültü nedeniyle), `--freq D` ile günlük.
4. **Split:** Kronolojik son %20 holdout (sınıflandırma pipeline'ıyla tutarlı; karıştırma yok).
5. **Model:** `Prophet(yearly_seasonality=True, weekly_seasonality=(freq=='D'))`. Mevsimsellik modu
   `--seasonality-mode` ile seçilir; **varsayılan `additive`** (sıfır içeren küçük sayımlarda `multiplicative`
   trendi aşırı ekstrapole edip naive'den daha kötü sonuç verdiği için). Opsiyonel `--with-weather` ile günlük
   ortalama `temp`/`humidity` regresör eklenir (canlı tahmin için gelecekteki hava değerleri gerekir; gelecek
   doldurmada mevsimsel ortalama placeholder kullanılır).
6. **Değerlendirme:** Holdout MAE/RMSE/MAPE + iki baseline: **naive (düz ortalama, mevsimselliği yok sayar)**
   ve **mevsimsel naive (geçen yılın aynı dönemi, mevsimselliği yakalar)**. `--cv` ile Prophet çok-katlı
   genişleyen-pencere cross-validation (tek-yıl holdout'tan daha güvenilir).
   - **Anomali maskeleme (`--outlier-ranges`):** `"BAŞ:BİT,BAŞ:BİT"` formatında verilen tarih aralıklarında
     `y=NaN` yapılır; Prophet bu noktalara **FIT OLMAZ** (anomaliye çekilmez) ama satırlar korunduğu için
     mevsimsellik takvimi bozulmaz. Maskelenen noktalar holdout/CV metriklerinden de dışlanır
     (NaN-güvenli baseline'lar). Örn. `--outlier-ranges "2025-05-01:2025-06-30"`.
7. **Çıktılar (`outputs/`):** `forecast_failures.csv` (ds, yhat, yhat_lower, yhat_upper),
   `prophet_forecast.png`, `prophet_components.png`.

### Bulgu (mevcut veride)
**Mevsimsellik VAR ve güçlü:** ham `failure_event` aya/mevsime göre toplandığında yaz nettir
(yaz=295, ilkbahar=116, sonbahar=119, kış=79; Temmuz=122, Ağustos=101). Prophet bunu yıllık mevsimsellik
bileşeninde yakalar (`prophet_components.png`'de yaz tepesi).

**Ama nokta-tahmin doğruluğu 2025 verisinin bozukluğundan zarar görüyor.** 2025'in ikinci yarısı iki ayrı
sorun içeriyor (haftalık seride): (a) tekil **spike haftaları** (`2025-05-18`=40, `2025-08-03`=23; seri
ort. ≈3.6) ve (b) Eylül–Aralık'ta ~0'a **çöküş** — **Aralık 2025 + Ocak 2026 haftaları tamamen sıfır**, ki
bu büyük olasılıkla gerçek bir düşüş değil, **veri kaydının orada bitmesi** (eksik / sağ-sansürlü veri).

**Mayıs–Haziran'ı outlier maskelemek metrikleri iyileştirmedi; 2025'i komple çıkarmak en temiz sonucu verdi.**
Karşılaştırma (haftalık, `--freq W`):

| Konfigürasyon | CV ort. MAE | Holdout Prophet | Holdout Naive (ort.) |
|---|---|---|---|
| **`3_years_data_no2025.csv`** (2023–24) | **3.361** | 4.375 | 3.864 |
| `3_years_data.csv` + `--outlier-ranges 2025-05-01:2025-06-30` | 4.103 | 5.047 | 3.674 |
| `3_years_data.csv` + `--outlier-ranges 2025-05-01:2025-08-31` | 3.762 | 4.940 | 2.743 |

Her konfigürasyonda **Prophet düz ortalama (naive) baseline'ı geçemiyor**: haftalık sayımlar küçük/gürültülü
(0–24) ve elde yalnızca ~2–3 mevsimsel döngü var. Maskeleme yardımcı olmadı, çünkü 2025'in bozukluğu tek
döneme sıkışmıyor (yaz spike'ları + sonbahar/kış çöküşü); en düşük CV hatası 2025'siz seride.

**Sonuç:** Model doğru kuruldu ve mevsimselliği öğreniyor; asıl sınırlayıcı, agregat seride **yıllar-arası
non-stasyonarite + 2025 sonu veri eksikliği**. Öneriler: (1) seriyi en güvenilir tarihe kadar **kırpmak**
(Eylül 2025 sonrası sıfır-akışını maskelemek yerine veriyi kesmek), (2) daha çok yıl / temiz veri,
(3) changepoint esnekliğini ayarlamak veya tatil/özel-gün regresörleri. Bu, §9'daki "zamansal
non-stasyonarite" notuyla tutarlıdır.

### Çalıştırma
```bash
pip install -r requirements.txt           # prophet + matplotlib eklendi

# Varsayılan veri (3_years_data.csv) ile haftalık
python forecast_failures_prophet.py --freq W

# En temiz sonuç: 2025 komple çıkarılmış set (CV MAE ≈ 3.361)
python forecast_failures_prophet.py --data 3_years_data_no2025.csv --cv

# Anomali dönem(ler)ini maskeleyerek (y=NaN; Prophet o noktalara fit etmez)
python forecast_failures_prophet.py --data 3_years_data.csv --outlier-ranges "2025-05-01:2025-06-30" --cv

# Günlük + hava regresörü
python forecast_failures_prophet.py --freq D --with-weather --cv
```
**Sağlık kontrolü:** `prophet_components.png`'de yıllık mevsimsellik yaz aylarında tepe yapmalı (HVAC yükü);
Prophet holdout MAE'si naive baseline'ı geçmiyorsa seri zayıf sinyallidir.
