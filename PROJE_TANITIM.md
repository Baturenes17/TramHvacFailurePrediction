# Tram HVAC Arıza Tahmini — Proje Tanıtımı

Bu belge, projenin **ne yaptığını**, **hangi veriyi kullandığını**, **hangi sınıfları tahmin ettiğini** ve **hangi modellerle çalıştığını** sade bir şekilde özetler. Yeni bir projeye başlarken bağlam vermek için kullanılabilir.

---

## 1. Amaç

Tramvaylardaki **HVAC (klima/iklimlendirme) sistemi arızalarını önceden tahmin etmek**.

Model, her aracın günlük kayıtlarını analiz eder ve o aracın **yakın gelecekte arıza yapma olasılığını** üretir. Böylece bakım ekibi arıza olmadan önce müdahale edebilir (kestirimci bakım → beklenmeyen arıza ve duruş süresi azalır).

---

## 2. Problem Tipi

**İkili (binary) sınıflandırma** — bir araç-gün kaydı için "arıza olacak (1)" / "arıza olmayacak (0)".

İki ayrı tahmin ufku (horizon) var:

| Hedef sütun | Anlamı | Sınıflar |
|---|---|---|
| `failure_next_7d` (veya eski `label_7d`) | Önümüzdeki **7 gün** içinde HVAC arızası olacak mı? | 0 = arıza yok, 1 = arıza var |
| `failure_next_30d` (veya eski `label_30d`) | Önümüzdeki **30 gün** içinde HVAC arızası olacak mı? | 0 = arıza yok, 1 = arıza var |

> Aynı anda yalnızca bir ufuk eğitilir (`PREDICTION_HORIZON_DAYS = 7` veya `30`). Sızıntıyı (leakage) önlemek için diğer ufkun etiketi özelliklerden çıkarılır.

> **Sınıf dengesizliği:** Arıza (sınıf 1) nadirdir; veri çoğunlukla "arıza yok" (sınıf 0) satırlarından oluşur. Bu yüzden tek başına accuracy yanıltıcıdır; ROC-AUC, PR-AUC ve precision/recall daha anlamlıdır.

---

## 3. Veri Yapısı

Her satır bir **araç-gün** gözlemidir. Temel sütunlar:

- **Kimlik:** `date`, `vehicle_id`, `vehicle_type`, `cabin_id`
- **Kullanım:** `km_today`, `km_last_7d`, `km_last_30d`, `vehicle_age`
- **Bakım/arıza geçmişi:** `days_since_last_maintenance`, `km_since_last_maintenance`, `days_since_last_failure`, `other_failures_since_last_maintenance`, `failures_last_90d`
- **Hava durumu:** `temp`, `humidity`, `wind_speed`, `weather_type`
- **Hedef etiketler:** `failure_next_7d`, `failure_next_30d` (eski şemada `label_7d`, `label_30d`)

Etiketler, satırın tarihinden sonraki ilgili pencerede (7/30 gün) gerçek bir HVAC arızası olup olmadığına bakılarak üretilir.

---

## 4. Özellik Üretimi (Feature Engineering)

Tümü **zaman-bilinçli ve sızıntısız** (gelecekteki etiketten asla türetilmez):

- **Takvim/mevsim:** ay, haftanın günü, gün-içi-yıl sinüs/kosinüs, mevsim, yaz bayrağı
- **Hava-stres:** `temp²`, sıcak/soğuk bayrağı, sıcaklık×nem, aylık iklim ortalamasından sapma
- **Araç geçmişi (causal):** geçmiş arıza sayısı, geçmiş arıza oranı, son 90 gündeki arızalar (hepsi `shift(1)` ile, yani sadece geçmiş bilgisi)
- **Kullanım trendi:** 7g/30g km oranı, km'nin 30 günlük yuvarlanan std'si

---

## 5. Kullanılan Modeller

Tablo tipi, zamana bağlı veride güçlü olan **ağaç tabanlı** modeller:

- **LightGBM** (varsayılan)
- **XGBoost**
- **CatBoost**
- **Ensemble** — yukarıdaki üç modelin olasılık ortalaması (soft voting)

> Not: Bu veri setinde tek modelin AUC tavanı ~0.73 civarında; CatBoost/ensemble en etkili kaldıraç (~0.76'ya kadar). Ek özellikler genelde belirgin katkı sağlamıyor.

Ek yetenekler:
- **Optuna** ile (zaman bazlı expanding-window CV üzerinde PR-AUC maksimize ederek) LightGBM hiperparametre araması
- **SHAP** ile özellik önem analizi

---

## 6. Eğitim & Değerlendirme Yaklaşımı

- **Zaman bazlı bölme:** train %60 / validation %20 / test %20 — tarihe göre sıralı (gelecekten geçmişe sızıntı yok)
- **Önişleme:** sayısallarda medyan doldurma; kategoriklerde mod doldurma + ordinal encoding
- **Eşik seçimi (iki mod):**
  - **Alarm-oranı (varsayılan):** her dönemde skorca en riskli %k araç pozitif işaretlenir. Base-rate kayması olsa bile val/test precision'ını dengeler ("her dönem en riskli %k aracı incele" → bakım kapasitesine doğal uyum).
  - **Sabit eşik:** validation'da recall ≥ 0.80 kısıtı altında precision-maksimum eşik.

**Raporlanan metrikler:** ROC-AUC, PR-AUC (average precision), precision/recall/F1 ve recall hedeflerine göre precision-recall ödünleşim tablosu.

---

## 7. Çıktılar

Her araç-gün kaydı için:
- 7 gün / 30 gün içinde HVAC arıza olasılığı
- Risk skoruna göre sıralanmış araç listesi (önceliklendirme için)
- Tahmini yönlendiren özellikler (SHAP)

---

## 8. Çalıştırma (Örnek)

```bash
# Varsayılan: LightGBM, alarm-oranı %40
python train_failure_model.py --data 2024_full_future_labels.csv

# Ensemble + SHAP
python train_failure_model.py --model ensemble --shap

# Optuna ile hiperparametre araması
python train_failure_model.py --model lightgbm --tune --trials 40
```

Ufku değiştirmek için kod içindeki `PREDICTION_HORIZON_DAYS` değerini `7` ya da `30` yap.
