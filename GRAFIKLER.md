# GRAFİKLER — Komut Çıktıları

Bu belge, çalıştırılan 5 komutun grafiklerini ve **hangi grafiğin hangi komuta ait**
olduğunu açıklar. Tüm grafikler `outputs/grafikler/` klasöründedir.

> **Önemli not:** `train_failure_model.py` (Komut 1–3) varsayılan olarak grafik üretmez
> (sadece CSV + metin metrik verir). Bu yüzden o komutlar için rapora konulabilecek standart
> **değerlendirme grafiklerini** (ROC eğrisi, Precision-Recall eğrisi, Confusion matrix)
> `make_graphs.py` ile, komutun pipeline'ı **birebir** tekrarlanarak ürettik (aynı model, aynı
> fixed eşik). `forecast_failures_prophet.py` (Komut 4–5) ise grafikleri **kendisi** üretir.

---

## Komut 1 — `3_years_data_no7d.csv` (dengeleme yok)
```
python train_failure_model.py --data 3_years_data_no7d.csv --model lightgbm --test-end 2025-12-01 --threshold-mode fixed
```
**Grafik:** [outputs/grafikler/cmd1_no7d.png](outputs/grafikler/cmd1_no7d.png)

Tek görselde 3 panel:
- **Sol — ROC Eğrisi (Train + Test):** Train AUC≈0.948, Test AUC≈0.711. İki eğri arasındaki
  büyük açıklık **overfit'i** gösterir (ağaç modeli train'i ezberliyor).
- **Orta — Precision-Recall Eğrisi (Test):** PR-AUC (AP)≈0.336. Kesik gri çizgi baz oranı
  (pozitif sınıf yüzdesi ≈0.19); eğri ne kadar yüksekse model o kadar iyi.
- **Sağ — Confusion Matrix (Test, fixed eşik≈0.599):** Gerçek vs tahmin sayıları (TN/FP/FN/TP).

---

## Komut 2 — `3_years_data_no7d_undersampled.csv`
```
python train_failure_model.py --data 3_years_data_no7d_undersampled.csv --model lightgbm --test-end 2025-12-01 --threshold-mode fixed
```
**Grafik:** [outputs/grafikler/cmd2_undersampled.png](outputs/grafikler/cmd2_undersampled.png)

3 panel (yukarıdakiyle aynı düzen): Test ROC-AUC≈0.687, **PR-AUC≈0.645**. Undersample edilmiş
sette test de dengelendiği için (baz oran≈0.48) PR-AUC ve confusion matrix çok daha dengeli
görünür — ama bu, gerçek operasyon dağılımını yansıtmaz (optimist).

---

## Komut 3 — `3_years_data_no7d_undersampled.csv` + SMOTE
```
python train_failure_model.py --data 3_years_data_no7d_undersampled.csv --model lightgbm --test-end 2025-12-01 --threshold-mode fixed --smote
```
**Grafik:** [outputs/grafikler/cmd3_undersampled_smote.png](outputs/grafikler/cmd3_undersampled_smote.png)

3 panel. Test ROC-AUC≈0.689, PR-AUC≈0.641. Komut 2 ile neredeyse aynı — çünkü undersample edilmiş
veri zaten ~1:1 dengede olduğundan SMOTE ek sentetik örnek (neredeyse) üretmiyor. **SMOTE bu veride
fark yaratmıyor**; grafikler de bunu görsel olarak doğruluyor.

---

## Komut 4 — Prophet (`3_years_data_no2025.csv`)
```
python forecast_failures_prophet.py --data 3_years_data_no2025.csv
```
**Grafikler (2 adet, Prophet'in kendi çıktısı):**
- [outputs/grafikler/cmd4_prophet/prophet_forecast.png](outputs/grafikler/cmd4_prophet/prophet_forecast.png)
  — **Tahmin grafiği:** Siyah noktalar gerçek haftalık arıza sayısı; mavi çizgi Prophet tahmini;
  açık mavi bant %80 güven aralığı. Sağ uçtaki çizgi geleceğe (8 hafta) uzanan tahmindir.
- [outputs/grafikler/cmd4_prophet/prophet_components.png](outputs/grafikler/cmd4_prophet/prophet_components.png)
  — **Bileşen grafiği:** Prophet'in ayrıştırdığı **trend** + **yıllık mevsimsellik**. Yıllık
  bileşende **yaz tepesi** beklenir (HVAC yükü) — mevsimselliğin yakalandığını gösterir.

---

## Komut 5 — Prophet + Cross-Validation (`--cv`)
```
python forecast_failures_prophet.py --data 3_years_data_no2025.csv --cv
```
**Grafikler:**
- [outputs/grafikler/cmd5_prophet/prophet_forecast.png](outputs/grafikler/cmd5_prophet/prophet_forecast.png)
  — Tahmin grafiği (Komut 4 ile aynı; `--cv` final tahmini değiştirmez, sadece ek doğrulama yapar).
- [outputs/grafikler/cmd5_prophet/prophet_components.png](outputs/grafikler/cmd5_prophet/prophet_components.png)
  — Bileşen grafiği (trend + yıllık mevsimsellik).
- [outputs/grafikler/cmd5_prophet_cv_mae.png](outputs/grafikler/cmd5_prophet_cv_mae.png)
  — **CV özel grafiği (bu komuta özgü):** Cross-validation hatasının (MAE) tahmin **ufkuna göre**
  değişimi. Kesik gri çizgi ortalama MAE≈3.36. Kısa ufuklarda hata düşük, bazı uzun ufuklarda
  yükselir — `--cv`'nin asıl kattığı bilgi budur (tek-yıl holdout'tan daha güvenilir).

---

## Özet Eşleme Tablosu

| Komut | Grafik dosyası | Ne gösterir |
|---|---|---|
| 1 | `cmd1_no7d.png` | ROC + PR + Confusion (dengeleme yok) |
| 2 | `cmd2_undersampled.png` | ROC + PR + Confusion (undersample) |
| 3 | `cmd3_undersampled_smote.png` | ROC + PR + Confusion (undersample + SMOTE) |
| 4 | `cmd4_prophet/prophet_forecast.png` | Filo arıza tahmini + güven aralığı |
| 4 | `cmd4_prophet/prophet_components.png` | Trend + yıllık mevsimsellik |
| 5 | `cmd5_prophet/prophet_forecast.png` | Tahmin (Komut 4 ile aynı) |
| 5 | `cmd5_prophet/prophet_components.png` | Bileşenler (Komut 4 ile aynı) |
| 5 | `cmd5_prophet_cv_mae.png` | CV: MAE vs tahmin ufku |

> Grafikleri yeniden üretmek için: `python make_graphs.py` (Komut 1–3 + CV grafiği) ve
> Prophet `--out` ile ayrı klasörlere (Komut 4–5). Sonuç sayıları için bkz.
> [PROJE_NIHAI_SONUCLAR.md](PROJE_NIHAI_SONUCLAR.md).
