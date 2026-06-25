# PROJE NİHAİ KOD — Teknik Uygulama Detayları

Bu belge, projeyi **kod tarafında nasıl yaptığımızı** anlatır: hangi kütüphaneleri/teknikleri
**ne için, nasıl ve neden** kullandık. Her bölümde ilgili kod parçası + gerekçe vardır.

> Sonuç çıktıları için bkz. [PROJE_NIHAI_SONUCLAR.md](PROJE_NIHAI_SONUCLAR.md)
> Genel pipeline dokümantasyonu için bkz. [PROJE.md](PROJE.md)

---

## 0. Dosya / Script Haritası

| Dosya | Görev |
|---|---|
| `train_failure_model.py` | **Ana pipeline** — araç-bazlı 30 günlük arıza sınıflandırması (eğitim + tahmin) |
| `forecast_failures_prophet.py` | Filo geneli arıza **sayısı** zaman serisi tahmini (Prophet) |
| `undersample_data.py` | `3_years_data_no7d.csv` → 1:1 dengeli `..._undersampled.csv` üretir |
| `feature_importance.py` | LightGBM gain-bazlı özellik önemi (ana pipeline fonksiyonlarını import eder) |
| `benchmark_models.py` | Birden çok modeli aynı split'te karşılaştırır |
| `diagnose_and_cv.py` | Zaman-serisi CV teşhisi |
| `precision_target_analysis.py` | Precision hedefli eşik analizi |
| `final_operating_point.py` | Operasyonel çalışma noktası seçimi |

### Kullanılan kütüphaneler ve **neden**
| Kütüphane | Ne için | Neden bu seçildi |
|---|---|---|
| `pandas` / `numpy` | Veri yükleme, feature engineering, vektörel hesap | Tablo + zaman serisi işleme standardı |
| `lightgbm` | Ana sınıflandırıcı | Dengesiz tabular veride hızlı, güçlü; `scale_pos_weight` desteği |
| `scikit-learn` | Önişleme (`ColumnTransformer`, `SimpleImputer`, `OrdinalEncoder`), metrikler, `Pipeline` | Sızıntısız, tek paket halinde eğitim/tahmin tutarlılığı |
| `xgboost`, `catboost` | Alternatif boosting modelleri | Karşılaştırma (model çeşitliliği test edildi) |
| `imbalanced-learn` | `SMOTENC` + `RandomUnderSampler` | Sınıf dengesizliğini sadece TRAIN'de düzeltmek |
| `optuna` | Hiperparametre araması | Zaman-serisi CV ile PR-AUC maksimizasyonu |
| `shap` | Özellik önemi açıklaması | Ağaç modelleri için `TreeExplainer` |
| `prophet` | Zaman serisi tahmini | Trend + yıllık mevsimsellik (yaz HVAC yükü) için ideal |
| `matplotlib` | Grafikler | Prophet/SHAP görselleri |

---

## 1. Veri Yükleme — `load_data()`

```python
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["vehicle_id", "date"]).reset_index(drop=True)
    return df
```

- **Nasıl:** CSV `;` ayraçlı ve ondalık `,` (Türkçe/Avrupa formatı) → `sep=";", decimal=","`.
- **Neden `vehicle_id, date` sıralaması:** Tüm araç-bazlı yuvarlanan istatistikler (`shift`, `rolling`)
  bu sıraya bağlıdır; doğru sıralama olmadan geçmiş/gelecek karışır → **sızıntı** olur.

---

## 2. Feature Engineering — `engineer_features()` (sızıntısız / causal)

**Temel kural:** Her türetilen özellik **yalnızca geçmiş veya aynı-gün** bilgisinden gelir;
gelecekteki etiketten asla türetilmez. Bu, modelin gerçekte bilemeyeceği bir bilgiyle "kopya
çekmesini" (data leakage) engeller.

### 2.1 Takvim / mevsim
```python
df["month"] = df["date"].dt.month
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
df["season"] = (df["month"] % 12 // 3).map({0:"winter",1:"spring",2:"summer",3:"autumn"})
```
- **Neden sin/cos:** Yıl-içi-gün döngüseldir (31 Aralık ile 1 Ocak komşudur). Ham sayı bunu
  bilmez; sin/cos kodlama döngüselliği modele öğretir.

### 2.2 Hava-stres
```python
df["temp_sq"] = df["temp"] ** 2          # doğrusal olmayan ısı stresi
df["is_hot"] = (df["temp"] >= 30).astype(int)
df["temp_x_humidity"] = df["temp"] * df["humidity"]   # etkileşim
month_mean_temp = df.groupby("month")["temp"].transform("mean")
df["temp_dev_from_month"] = df["temp"] - month_mean_temp  # iklim sapması
```
- **Neden `temp_sq` / etkileşim:** HVAC arıza riski sıcaklıkla doğrusal değil; aşırı sıcakta
  kuadratik artar. `temp_x_humidity` "bunaltıcı" günleri yakalar.
- **Neden `temp_dev_from_month`:** Mutlak sıcaklık yerine "mevsim normaline göre ne kadar sıcak"
  daha bilgilendirici; exogen (etiketten türetilmez) bir iklim sinyali.

### 2.3 Arıza geçmişi — causal türetme (kritik nokta)
Ham veride "bugün arıza oldu" sütunu **yok**. `days_since_last_failure` sayacı bir arıza olunca
sıfırlanır; o yüzden bu sayaç **bir önceki güne göre düştüğünde** o gün arıza olmuştur:

```python
dsf_prev = df.groupby("vehicle_id")["days_since_last_failure"].shift(1)
df["failure_event"] = (df["days_since_last_failure"] < dsf_prev).astype(int)
fe = df.groupby("vehicle_id")["failure_event"]
# Hepsi shift(1): bugünün/geleceğin arızası SAYIMA GİRMEZ -> sızıntı yok
df["veh_past_failure_rate"] = (veh_past_failures / veh_past_obs)   # durağan oran
df["veh_failures_last_90d"] = fe.transform(
    lambda s: s.shift(1, fill_value=0).rolling(90, min_periods=1).sum())
```
- **Neden `shift(1)`:** Bugünün arızasını özellik olarak kullanmak doğrudan hedefi sızdırırdı.
- **Neden kümülatif sayımlar (`veh_past_failures`, `veh_past_obs`) ÖZELLİK olarak DÜŞÜRÜLDÜ:**
  Zamanla monoton büyürler → bir tür "zaman indeksi" gibi davranırlar; takvim-bazlı split'te
  train→test dağılım kayması yaratıp test AUC'sini düşürürler (deneyle: 0.73 → 0.68). Sadece
  **durağan** `rate` ve `last_90d` tutuldu.

### 2.4 Birikimli hava stresi (tarih bazlı)
```python
daily_w = df.drop_duplicates("date").set_index("date").sort_index()["temp"]
roll["temp_7d_mean"] = daily_w.rolling(7, min_periods=1).mean()
roll["hot_days_7d"]  = (daily_w >= 30).rolling(7, min_periods=1).sum()
df = df.merge(roll, left_on="date", right_index=True, how="left")
```
- **Neden tek tarih serisi üzerinden:** Hava tüm filoda aynı; tarih başına bir kez hesaplayıp
  birleştirmek hem doğru hem hızlı. Pencereler aynı-gün dahil; çünkü **bugünün hava durumu
  gözlemlenmiştir** (sızıntı değil).

### 2.5 Hava tipi (WMO kodu) → anlamlı bayraklar
```python
df["is_precip"] = (wt >= 51).astype(int)
df["is_rain"]   = wt.between(61, 69).astype(int)
df["is_snow"]   = wt.between(71, 79).astype(int)
```
- **Neden:** WMO kodu ordinal değil kategorik; ham sayı olarak vermek yanlış sıralama varsayar.
  Anlamlı bayraklara çevirmek modele doğru bilgiyi verir.

### 2.6 Sızıntı doğrulaması
```python
exclude = set(ID_COLS) | {TARGET, OTHER_LABEL}   # date, vehicle_id, failure_next_30d, failure_next_7d
feature_cols = [c for c in df.columns if c not in exclude]
assert TARGET not in feature_cols and OTHER_LABEL not in feature_cols, "Sızıntı!"
```
- **Neden `failure_next_7d` çıkarılır:** Hedefle (`failure_next_30d`) güçlü korelasyonludur;
  bırakılırsa model neredeyse kopya çeker. (Bu deneylerde zaten `..._no7d.csv` setleri kullanıldı.)

**Toplam: 27 özellik.**

---

## 3. Önişleme — `build_preprocessor()`

```python
numeric_tf = SimpleImputer(strategy="median")
categorical_tf = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
ColumnTransformer([("num", numeric_tf, num), ("cat", categorical_tf, cat)], remainder="drop")
```
- **Nasıl:** Sayısal → medyan imputasyon; kategorik (`vehicle_type, weather_type, season,
  cevre_temizligi`) → en sık değerle doldur + ordinal encode.
- **Neden `OrdinalEncoder` (one-hot değil):** Ağaç modelleri ordinal kodlamayla iyi çalışır;
  one-hot gereksiz boyut artışı yapardı.
- **Neden `handle_unknown=use_encoded_value, unknown_value=-1`:** Tahmin/skor aşamasında train'de
  görülmemiş bir kategori gelirse hata vermeden `-1`'e maplenir (robustluk).
- **Neden ölçekleme yok:** Ağaç tabanlı model ölçeklemeye duyarsız. (Sadece `logreg` seçilirse
  `build_model` içinde `StandardScaler` eklenir.)
- **Neden tek `ColumnTransformer`/`Pipeline`:** `fit_transform(train)` → `transform(test)` aynı
  dönüşümü garanti eder; eğitim ve tahminde tutarlılık (sızıntısız).

---

## 4. Takvim Bazlı Bölme — `time_based_split()`

```python
dmin, dmax = df_sorted["date"].min(), df_sorted["date"].max()
train_end = dmin + (dmax - dmin) * TRAIN_FRAC          # zaman ekseninin ilk %80'i
train = df_sorted[df_sorted["date"] <= train_end]
test  = df_sorted[df_sorted["date"] >  train_end]
if test_end is not None:
    test = test[test["date"] <= test_end]              # --test-end
```
- **Neden satır sayısına göre değil, TARİH aralığına göre:** Gerçek hayatta geçmişle eğitip
  geleceği tahmin ederiz. Rastgele bölme gelecekten geçmişe sızıntı yapardı.
- **Neden `--test-end 2025-12-01`:** Verinin son ~30 gününde `failure_next_30d` etiketi tam
  30 günlük ileri pencereye sahip olmadığından **eksik/güvenilmezdir** (sağ-sansür). Bu kuyruk
  test'ten kırpılır ki metrikler bozuk etiketlerden etkilenmesin.

---

## 5. Model Kurulumu — `build_model()`

```python
def _build_lightgbm(scale_pos_weight, params=None):
    base = dict(n_estimators=200, learning_rate=0.03, num_leaves=31, max_depth=-1,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1)
    if params: base.update(params)
    return LGBMClassifier(**base)
```
- **`--model` seçenekleri:** `lightgbm` (varsayılan) | `logreg` | `xgboost` | `catboost` | `ensemble`
  (soft-voting). Bu deneylerde **lightgbm** kullanıldı.
- **Sınıf dengesizliği — `scale_pos_weight`:**
  ```python
  def compute_scale_pos_weight(y):
      return (y == 0).sum() / max(y.sum(), 1)   # negatif / pozitif
  ```
  - **Neden:** Pozitif (arıza) sınıf ~%20; model çoğunluğa kayar. `scale_pos_weight` pozitif
    örneklere kayıp fonksiyonunda daha fazla ağırlık verir.
- **Neden sabit 200 ağaç (early stopping varsayılan KAPALI):** Erken durdurma modeli çok erken
  (~5 ağaç) durdurup AUC'yi ve precision'ı düşürüyordu, skorları kabalaştırıp eşik kontrolünü
  bozuyordu. Deneyle en iyi bant 50–200 ağaç çıktı.
- **Neden `subsample`/`colsample`/`reg_lambda`:** Overfit'i sınırlamak için (ağaç modelleri bu
  veride train≫test farkıyla overfit eğilimli).

---

## 6. Sınıf Dengeleme Teknikleri (sadece TRAIN'e)

> **Altın kural:** Dengeleme **yalnızca train'e** uygulanır; test gerçek dağılımıyla kalır.
> Aksi halde test metrikleri yapay olarak iyi görünür (sızıntı/kendini kandırma).

### 6.1 Önceden undersampling — `undersample_data.py`
Bu, ham CSV seviyesinde (pipeline'dan ayrı) çalışan bir yardımcıdır:
```python
keep_neg = min(len(neg_idx), RATIO * n_pos)   # RATIO=1 -> 1:1 tam denge
neg_sampled = set(random.sample(neg_idx, keep_neg))
kept = set(pos_idx) | neg_sampled             # orijinal satır sırasını koru
```
- **Ne yapar:** Çoğunluğu (failure=0) rastgele azaltıp azınlığa eşitler → `..._undersampled.csv`.
- **Neden satır-bazlı (pandas değil):** Orijinal metin formatını (`;`, ondalık `,`, tarih) birebir
  korumak için ham satırlar yeniden yazılmadan kopyalanır.
- **Önemli uyarı (sonuçlarda görüldü):** Bu undersampling **tüm veriye** (train+test) uygulandığı
  için test seti de dengelenmiş olur — bu yüzden undersampled set sonuçlarındaki yüksek precision/
  recall optimisttir; ROC-AUC ölçütü daha adildir.

### 6.2 SMOTE (pipeline içi, `--smote`) — `SMOTENC`
```python
n_num = len([c for c in feature_cols if c not in CATEGORICAL_FEATURES])
cat_idx = list(range(n_num, n_num + n_cat))   # encode'lu kategorikler en sonda
sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=strat, random_state=42)
Xtr, ytr = sm.fit_resample(Xtr, ytr)          # SADECE train
```
- **Neden `SMOTENC` (düz SMOTE değil):** Veride kategorik kolonlar var; SMOTENC kategorikleri
  doğru ele alır (komşuların modunu alır, interpolasyon yapmaz).
- **Neden preprocessor'dan SONRA:** SMOTENC NaN kabul etmez; önce `SimpleImputer` ile eksikler
  dolduruldu. `ColumnTransformer` çıktısında kolon sırası `[sayısal..., kategorik...]` olduğundan
  `cat_idx` en sondaki indekslerdir.
- **Neden SMOTE sonrası `scale_pos_weight` yeniden hesaplanır:** SMOTE dengeyi düzelttiğinde hem
  oversampling hem `scale_pos_weight` uygulamak **çifte düzeltme** olurdu → SMOTE sonrası y'den
  yeniden hesaplanır (denge 1:1 olunca spw≈1).

### 6.3 SMOTE + hafif undersampling (`--smote --undersample`)
```python
us = RandomUnderSampler(sampling_strategy=args.undersample_ratio, random_state=42)
Xtr, ytr = us.fit_resample(Xtr, ytr)
```
- **Fikir:** Önce SMOTE azınlığı ara dengeye (`--smote-ratio`) çıkarır, sonra çoğunluk kısılarak
  hedef min:maj oranına (`--undersample-ratio`) ulaşılır. Tam 1:1 yerine **ılımlı denge** çoğu
  zaman daha iyi genelleme verir. Tutarlılık kontrolü: `undersample-ratio > smote-ratio` olmalı.

> **Sonuçlardaki bulgu:** Bu veride SMOTE faydalı olmadı — tam veride recall'ı çökertti, undersampled
> veride zaten denge olduğundan +0 sentetik üretti. Yani dengeleme darboğazı çözmüyor; sorun
> **veri/sinyal** tükenmesi.

---

## 7. Eşik Seçimi — 3 Mod

Model olasılık üretir; "alarm" kararı için bir eşik gerekir. Eşik **train üzerinden** seçilir
(test'e bakarak eşik seçmek sızıntı olurdu).

```python
def recall_constrained_threshold(y_true, scores, target_recall):
    # recall >= hedef altında precision'ı maksimize eden en yüksek eşik
def alarm_rate_threshold(scores, alarm_rate):
    return np.quantile(scores, 1.0 - alarm_rate)   # en riskli %k
```

| Mod (`--threshold-mode`) | Mantık | Ne zaman |
|---|---|---|
| `alarm` (varsayılan) | Her split kendi skorlarının en riskli %k'sı (vars. %10) | Base-rate kaymasına dayanıklı; **önerilen** |
| `fixed` | Train'de `recall ≥ 0.80` kısıtı altında precision-maks eşik; aynı eşik test'e | Bu deneylerde kullanıldı |
| `precision` | Train'de `precision ≥ hedef` kısıtı altında recall-maks eşik | Precision öncelikliyse |

- **Neden bu deneylerde `fixed`:** Tüm koşuları sabit bir operasyonel kural (recall≥0.80) altında
  karşılaştırmak için. **Neden `alarm` genelde daha sağlam:** Skor dağılımı dönemler arası kaydığından
  sabit eşik transferi güvenilmez; alarm-oranı her split'te kendi quantile'ını kullanır.

---

## 8. Değerlendirme ve Raporlama

```python
roc = roc_auc_score(y_true, scores)
ap  = average_precision_score(y_true, scores)        # PR-AUC
print(classification_report(y_true, pred, digits=4))
```
- **Neden accuracy değil ROC-AUC + PR-AUC:** Dengesiz veride accuracy yanıltıcı (hep "0" derse
  %80 doğru). **PR-AUC** azınlık (arıza) sınıfına odaklanır; bu problemde asıl ölçüt.
- **Hem train hem test raporlanır:** Train↔test farkı overfit'i gösterir (ağaç modellerinde büyük).
- **Precision/Recall ödünleşim tablosu:** Farklı recall hedefleri (0.90…0.50) için ulaşılabilir
  precision'ı listeler → operasyonel karar (kaç alarm, kaç kaçan arıza) için.

---

## 9. Tahmin Çıktısı — `score_latest()`

```python
final_model.fit(final_prep.fit_transform(df[feature_cols]), df[TARGET])  # TÜM veriyle
latest = full_df.sort_values("date").groupby("vehicle_id").tail(1)        # her aracın son günü
latest["failure_prob_30d"] = model.predict_proba(...)[:, 1]
latest = latest.sort_values("failure_prob_30d", ascending=False)          # riske göre sırala
out.to_csv("outputs/predictions_latest.csv", sep=";", decimal=",")
```
- **Neden tüm veriyle yeniden fit:** Değerlendirme bittikten sonra final model **tüm veriyi**
  görsün ki güncel skorlama en zengin bilgiyle yapılsın.
- **Neden her aracın son günü:** "Gelecek tahmini" pratikte = her aracın en güncel durumunu skorlamak
  → riske göre sıralı bakım önceliği listesi (80 araç, 1'er satır).

---

## 10. Prophet Pipeline — `forecast_failures_prophet.py`

Bu, **farklı bir iş sorusunu** çözer: "Filo genelinde önümüzdeki dönemde **kaç** arıza beklenir?"
(bakım kapasitesi planlaması). Sınıflandırmanın yerine geçmez, tamamlar.

### 10.1 Arıza olayı türetme (sınıflandırmayla AYNI causal kural)
```python
dsf_prev = df.groupby("vehicle_id")["days_since_last_failure"].shift(1)
df["failure_event"] = (df["days_since_last_failure"] < dsf_prev).astype(int)
```
- **Neden `failure_next_*` etiketleri KULLANILMAZ:** Onlar ileriye-dönük etikettir, gerçek olay
  değil. Prophet gerçek geçmiş olay sayısını modeller.

### 10.2 Zaman serisine indirgeme
```python
daily = df.groupby("date")["failure_event"].sum()
daily = daily.reindex(pd.date_range(min, max, freq="D"), fill_value=0)  # eksik gün=0
ts = daily.resample("W").agg({"y": "sum"})                              # haftalık (varsayılan)
ts = ts.reset_index().rename(columns={"date": "ds"})                    # Prophet formatı: ds, y
```
- **Neden eksik günler 0 ile doldurulur:** Arıza olmayan gün = 0 arıza; Prophet kesintisiz takvim ister.
- **Neden haftalık (`--freq W`) varsayılan:** Günlük seri çok seyrek/gürültülü (çoğu gün 0); haftalık
  toplama sinyali netleştirir.

### 10.3 Model
```python
m = Prophet(yearly_seasonality=True, weekly_seasonality=(freq=="D"),
            seasonality_mode="additive", interval_width=0.80)
```
- **Neden `yearly_seasonality`:** Yaz HVAC yükü → güçlü yıllık mevsimsellik (veride doğrulandı:
  yaz=295, kış=79 olay).
- **Neden `additive` (multiplicative değil):** Küçük/sıfır içeren sayımlarda multiplicative trendi
  aşırı ekstrapole edip naive'den kötü sonuç veriyordu.

### 10.4 Değerlendirme — iki baseline ile adil kıyas
```python
naive_pred  = mean(train["y"])                       # mevsimselliği YOK SAYAR
snaive_pred = "geçen yılın aynı haftası" (period=52) # mevsimselliği YAKALAR
```
- **Neden iki baseline:** Prophet'in gerçekten ek değer kattığını kanıtlamak için. Mevsimsel
  naive'i geçemezse Prophet'in trend+mevsimsellik modeli işe yaramıyor demektir.
- **`--cv` (cross-validation):**
  ```python
  cv_df = cross_validation(model, initial=..., period=..., horizon=...)
  perf  = performance_metrics(cv_df)
  ```
  - **Neden:** Tek-yıl holdout şanslı/şanssız olabilir; genişleyen-pencere çok-katlı CV daha
    güvenilir ortalama MAE verir. (Sonuçlarda 2025'siz sette CV MAE 4.91 → 3.36'ya düştü.)

### 10.5 Neden 2025'siz set (`3_years_data_no2025.csv`)
2025'in ikinci yarısında (a) tekil spike haftaları, (b) Eylül–Aralık'ta ~0'a çöküş (büyük olasılıkla
**veri kaydının bitmesi** = sağ-sansür) var. Bu bozukluk Prophet'in nokta-tahminini bozuyordu;
2025 komple çıkarılınca en temiz sonuç alındı (CV MAE ≈ 3.36).

---

## 11. Yardımcı Script — `feature_importance.py`

Ana pipeline fonksiyonlarını **import ederek** (kod tekrarı yok) LightGBM gain-bazlı özellik önemi
çıkarır:
```python
from train_failure_model import (load_data, engineer_features, get_feature_columns,
    time_based_split, build_preprocessor, build_model, compute_scale_pos_weight)
gain = model.booster_.feature_importance(importance_type="gain")
```
- **Neden gain:** Bir özelliğin split'lerde sağladığı toplam kayıp azalması; "split sayısı"ndan
  daha anlamlı bir önem ölçüsü. Çıktı → `outputs/feature_importance.csv`.

---

## 12. Özet — Tasarım Felsefesi

1. **Sızıntısızlık her şeyin önünde:** `shift(1)`, takvim-bazlı split, `--test-end` kırpma,
   dengelemenin sadece train'e uygulanması, `assert` ile doğrulama.
2. **Dengesiz veriye uygun ölçüm:** accuracy değil ROC-AUC + PR-AUC + precision/recall ödünleşimi.
3. **Operasyonel gerçekçilik:** alarm-oranı eşiği + tüm veriyle final fit + araç önceliklendirme.
4. **Tek pipeline, tekrar yok:** Önişleme + model tek `Pipeline`/`ColumnTransformer`; yardımcı
   scriptler ana modülü import eder.
5. **Bulgu:** Darboğaz model veya teknik değil; **veri/sinyal** tükenmiş. Yeni özellikler ve
   dengeleme (SMOTE/undersample) precision+recall'ı birlikte yükseltmedi → gerçek iyileşme için
   yeni sinyal (HVAC telemetri/hata kodları) gerekir.
