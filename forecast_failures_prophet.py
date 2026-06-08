"""
Filo Geneli Arıza Tahmini — Facebook Prophet Zaman Serisi Pipeline'ı
====================================================================

Amaç: `train_failure_model.py` araç-bazlı İKİLİ SINIFLANDIRMA yaparken ("hangi araç
30 gün içinde arıza yapar?"), bu script TAMAMLAYICI bir soruya cevap verir:
"Önümüzdeki dönemde FİLO GENELİNDE kaç HVAC arızası beklenir?" (bakım kapasitesi /
personel planlaması). Bu, tek değişkenli bir zaman serisi tahmini problemidir ve
Prophet'in güçlü olduğu alandır (trend + yıllık/haftalık mevsimsellik).

Akış:
  1. Veriyi yükle (`;` ayraçlı, `,` ondalık) — train_failure_model.load_data ile aynı.
  2. Gerçek arıza olaylarını türet: `days_since_last_failure` sayacı düştüğü gün arıza
     olmuştur (train_failure_model.py:116-118 ile aynı causal kural). NOT: ileriye-dönük
     `failure_next_30d`/`failure_next_7d` ETİKETLERİ kullanılmaz.
  3. Tarihe göre toplayıp tek bir zaman serisine indir (günlük; varsayılan haftalık).
  4. Kronolojik train/holdout böl (son ~%20). Karıştırma yok — sızıntı yok.
  5. Prophet eğit (yıllık + opsiyonel haftalık mevsimsellik, opsiyonel hava regresörü).
  6. Holdout üzerinde MAE / RMSE / MAPE; naive baseline ile kıyas. Opsiyonel CV.
  7. Geleceğe tahmin üret -> outputs/forecast_failures.csv + tahmin/bileşen grafikleri.

Kullanım:
  python forecast_failures_prophet.py --freq W
  python forecast_failures_prophet.py --freq D --with-weather --cv
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Konfigürasyon
# --------------------------------------------------------------------------- #
DEFAULT_DATA = "3_years_data.csv"
DEFAULT_OUT = "outputs"
TRAIN_FRAC = 0.80                 # Holdout = kalan %20 (kronolojik), train_failure_model ile tutarlı
RANDOM_STATE = 42
# Varsayılan gelecek ufku (periyod sayısı): haftalık -> 8 hafta, günlük -> 30 gün
DEFAULT_HORIZON = {"W": 8, "D": 30}
WEATHER_COLS = ["temp", "humidity"]


# --------------------------------------------------------------------------- #
# 1. Veri yükleme (train_failure_model.load_data ile aynı)
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["vehicle_id", "date"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 2. Gerçek arıza olaylarını türet (train_failure_model.py:116-118 ile aynı)
# --------------------------------------------------------------------------- #
def derive_failure_events(df: pd.DataFrame) -> pd.DataFrame:
    """`days_since_last_failure` bir arıza olduğunda sıfırlanır; önceki güne göre
    DÜŞTÜYSE o gün bir arıza olmuştur -> failure_event = 1."""
    df = df.copy()
    dsf_prev = df.groupby("vehicle_id")["days_since_last_failure"].shift(1)
    df["failure_event"] = (df["days_since_last_failure"] < dsf_prev).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 3. Filo-seviyesi zaman serisine indir
# --------------------------------------------------------------------------- #
def build_timeseries(df: pd.DataFrame, freq: str, with_weather: bool) -> pd.DataFrame:
    """Tarihe göre filo geneli günlük arıza sayısı; eksik günler 0 ile doldurulur.
    freq='W' ise haftalık toplama yapılır. Prophet formatı: ds, y (+ opsiyonel regresörler)."""
    # Günlük filo arıza sayısı
    daily = (
        df.groupby("date")["failure_event"].sum().rename("y").to_frame()
    )
    # Eksik günleri 0 ile tamamla (kesintisiz takvim)
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0)
    daily.index.name = "date"

    # Hava regresörleri: günlük filo ortalaması (exogen; etiketten türetilmez)
    if with_weather:
        weather = df.groupby("date")[WEATHER_COLS].mean()
        weather = weather.reindex(full_idx).interpolate().ffill().bfill()
        weather.index.name = "date"
        daily = daily.join(weather)

    if freq == "W":
        agg = {"y": "sum"}
        if with_weather:
            agg.update({c: "mean" for c in WEATHER_COLS})
        ts = daily.resample("W").agg(agg)
    else:
        ts = daily

    ts = ts.reset_index().rename(columns={"date": "ds"})
    return ts


def apply_outliers(ts: pd.DataFrame, ranges: str | None):
    """Verilen tarih aralıklarındaki y'yi NaN yapar. Prophet bu noktalara FIT OLMAZ
    (anomaliye çekilmez) ama takvim/mevsimsellik sürekliliği bozulmasın diye satırlar
    KORUNUR. Bu, Prophet'in tek-seferlik anomaliler için önerdiği yöntemdir.
    Format: 'BAŞ:BİT,BAŞ:BİT' örn '2025-05-01:2025-06-30'. Döner: (ts, maskelenen_periyot)."""
    if not ranges:
        return ts, 0
    ts = ts.copy()
    mask = pd.Series(False, index=ts.index)
    for r in ranges.split(","):
        start, end = r.split(":")
        mask |= (ts["ds"] >= pd.Timestamp(start.strip())) & (ts["ds"] <= pd.Timestamp(end.strip()))
    n = int(mask.sum())
    ts.loc[mask, "y"] = np.nan
    return ts, n


# --------------------------------------------------------------------------- #
# 4. Kronolojik bölme
# --------------------------------------------------------------------------- #
def time_based_split(ts: pd.DataFrame):
    """Son %20 dönemi holdout olarak ayır (kronolojik). Karıştırma yok -> sızıntı yok."""
    n = len(ts)
    cut = int(n * TRAIN_FRAC)
    return ts.iloc[:cut].copy(), ts.iloc[cut:].copy()


# --------------------------------------------------------------------------- #
# 5. Metrikler
# --------------------------------------------------------------------------- #
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # MAPE: y_true=0 noktalarını dışla (sıfıra bölme)
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100 if nz.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}


def seasonal_naive_pred(ts: pd.DataFrame, holdout: pd.DataFrame, freq: str) -> np.ndarray:
    """Mevsimsel naive: her holdout noktası için 'bir yıl önceki aynı dönem' değeri.
    Düz ortalama yerine bu, mevsimselliği YAKALAYAN adil bir baseline'dır — Prophet'in
    gerçekten ek değer katıp katmadığını test eder."""
    period = 52 if freq == "W" else 365  # bir yıllık geri kayma
    y_full = ts["y"].values
    fallback = np.nanmean(y_full)  # NaN-güvenli (outlier maskeleme ile uyumlu)
    n_train = len(ts) - len(holdout)
    preds = []
    for i in range(len(holdout)):
        idx = n_train + i - period
        # Yeterli geçmiş yoksa eldeki en eski mevsimsel değere geri düş
        while idx < 0:
            idx += period
        val = y_full[idx] if idx < len(y_full) else fallback
        # Geçen yılın aynı haftası outlier olarak maskelendiyse fallback kullan
        preds.append(fallback if np.isnan(val) else val)
    return np.asarray(preds, dtype=float)


# --------------------------------------------------------------------------- #
# 6. Ana akış
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filo geneli HVAC arıza sayısı tahmini (Facebook Prophet)."
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help="Girdi CSV yolu")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Çıktı dizini")
    parser.add_argument(
        "--vehicle", type=int, default=None,
        help="Sadece bu vehicle_id için tahmin (araç-bazlı). Belirtilmezse filo geneli. "
             "UYARI: tek araç serisi çok seyrek (sparse) olduğundan Prophet zayıf çalışır.",
    )
    parser.add_argument(
        "--freq", choices=["D", "W"], default="W",
        help="Zaman serisi frekansı: D=günlük, W=haftalık (varsayılan: W)",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Gelecek tahmin ufku (periyod). Varsayılan: haftalık=8, günlük=30",
    )
    parser.add_argument(
        "--seasonality-mode", choices=["additive", "multiplicative"], default="additive",
        help="Mevsimsellik modu. Küçük/sıfır içeren sayımlarda 'additive' daha sağlamdır "
             "(varsayılan: additive)",
    )
    parser.add_argument(
        "--with-weather", action="store_true",
        help="Günlük ortalama temp/humidity'yi Prophet regresörü olarak ekle "
             "(canlı tahmin için gelecekteki hava değerleri gerekir)",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Prophet genişleyen-pencere cross-validation çalıştır",
    )
    parser.add_argument(
        "--outlier-ranges", default=None,
        help="Anomali tarih aralıkları; bu periyodlarda y=NaN yapılır (Prophet fit etmez, "
             "ama mevsimsellik takvimi korunur). Format: 'BAŞ:BİT,BAŞ:BİT' "
             "örn '2025-05-01:2025-06-30'",
    )
    args = parser.parse_args()

    # Prophet/matplotlib'i burada import et ki --help bağımlılık olmadan çalışsın
    from prophet import Prophet

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    horizon = args.horizon if args.horizon is not None else DEFAULT_HORIZON[args.freq]

    # --- Veri & seri ---
    df = load_data(args.data)
    df = derive_failure_events(df)

    # Araç-bazlı mod: arıza olaylarını TÜM filoda türettikten sonra tek araca filtrele
    # (days_since_last_failure sayacı zaten araç-içi shift ile hesaplandı, sızıntı yok).
    if args.vehicle is not None:
        ids = sorted(int(v) for v in df["vehicle_id"].unique())
        if args.vehicle not in ids:
            raise SystemExit(
                f"vehicle_id={args.vehicle} veride yok. Mevcut ID örnekleri: "
                f"{ids[:10]} ... (toplam {len(ids)} araç)"
            )
        df = df[df["vehicle_id"] == args.vehicle].copy()

    total_failures = int(df["failure_event"].sum())
    ts = build_timeseries(df, freq=args.freq, with_weather=args.with_weather)
    ts, n_outlier = apply_outliers(ts, args.outlier_ranges)

    scope = f"ARAÇ {args.vehicle}" if args.vehicle is not None else "FİLO GENELİ"
    print("=" * 70)
    print(f"{scope} ARIZA TAHMİNİ — Prophet")
    print("=" * 70)
    print(f"Frekans               : {args.freq}  (D=günlük, W=haftalık)")
    print(f"Toplam arıza olayı    : {total_failures}")
    print(f"Seri uzunluğu         : {len(ts)} periyod "
          f"({ts['ds'].min().date()} - {ts['ds'].max().date()})")
    print(f"Periyod başına y (ort): {ts['y'].mean():.2f}  (min={ts['y'].min():.0f}, max={ts['y'].max():.0f})")
    if n_outlier:
        print(f"Outlier maskelenen    : {n_outlier} periyod (y=NaN -> Prophet fit etmez) "
              f"[{args.outlier_ranges}]")

    # --- Kronolojik split ---
    train, holdout = time_based_split(ts)
    print(f"\nTrain  : {len(train)} periyod ({train['ds'].min().date()} - {train['ds'].max().date()})")
    print(f"Holdout: {len(holdout)} periyod ({holdout['ds'].min().date()} - {holdout['ds'].max().date()})")

    # --- Model ---
    def make_model() -> "Prophet":
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=(args.freq == "D"),
            daily_seasonality=False,
            seasonality_mode=args.seasonality_mode,
            interval_width=0.80,
        )
        if args.with_weather:
            for c in WEATHER_COLS:
                m.add_regressor(c)
        return m

    model = make_model()
    model.fit(train)

    # --- Holdout değerlendirmesi ---
    holdout_future = holdout[["ds"] + (WEATHER_COLS if args.with_weather else [])].copy()
    holdout_fc = model.predict(holdout_future)
    yhat_holdout = np.clip(holdout_fc["yhat"].values, 0, None)  # negatif arıza sayısı olamaz

    # Outlier maskeli (y=NaN) holdout noktalarını değerlendirmeden dışla — adil kıyas için
    # üç model de aynı geçerli noktalarda ölçülür.
    y_true_h = holdout["y"].values
    valid = ~np.isnan(y_true_h)
    m_prophet = _metrics(y_true_h[valid], yhat_holdout[valid])

    # Baseline 1 — Naive (düz ortalama): mevsimselliği YOK SAYAR (NaN-güvenli ortalama)
    naive_pred = np.full(len(holdout), np.nanmean(train["y"]))
    m_naive = _metrics(y_true_h[valid], naive_pred[valid])

    # Baseline 2 — Mevsimsel naive (geçen yılın aynı dönemi): mevsimselliği YAKALAR.
    # Asıl adil kıyas budur; Prophet bunu geçerse mevsimsellik+trendden ek değer üretiyor demektir.
    snaive_pred = seasonal_naive_pred(ts, holdout, args.freq)
    m_snaive = _metrics(y_true_h[valid], snaive_pred[valid])

    print("\n" + "-" * 70)
    print("HOLDOUT DEĞERLENDİRME")
    print("-" * 70)
    print(f"{'Model':<22}{'MAE':>10}{'RMSE':>10}{'MAPE%':>10}")
    print(f"{'Prophet':<22}{m_prophet['MAE']:>10.3f}{m_prophet['RMSE']:>10.3f}{m_prophet['MAPE%']:>10.2f}")
    print(f"{'Naive (ortalama)':<22}{m_naive['MAE']:>10.3f}{m_naive['RMSE']:>10.3f}{m_naive['MAPE%']:>10.2f}")
    print(f"{'Mevsimsel naive':<22}{m_snaive['MAE']:>10.3f}{m_snaive['RMSE']:>10.3f}{m_snaive['MAPE%']:>10.2f}")
    best = min([("Prophet", m_prophet["MAE"]), ("Naive (ortalama)", m_naive["MAE"]),
                ("Mevsimsel naive", m_snaive["MAE"])], key=lambda x: x[1])
    print(f"-> En düşük MAE: {best[0]} ({best[1]:.3f}).")
    if best[0] == "Prophet":
        print("   Prophet her iki baseline'ı da geçti — mevsimsellik+trendden ek değer üretiyor.")
    elif best[0] == "Mevsimsel naive":
        print("   Mevsimsellik GÜÇLÜ (mevsimsel naive düz ortalamayı yener) ancak bu tek-yıl")
        print("   holdout'unda Prophet onu geçemedi; yıllar arası genlik/zamanlama kayması etkili.")
        print("   Daha sağlam karşılaştırma için --cv (çok-katlı) önerilir.")
    else:
        print("   Mevsimsel naive bile düz ortalamayı geçemedi — bu holdout penceresinde")
        print("   yıllar arası patern kaymış olabilir.")

    # --- Opsiyonel cross-validation ---
    if args.cv:
        from prophet.diagnostics import cross_validation, performance_metrics
        unit = "D" if args.freq == "D" else "W"
        span = (ts["ds"].max() - ts["ds"].min()).days
        # makul pencere boyutları (gün cinsinden)
        initial = f"{int(span * 0.5)} days"
        period = f"{max(int(span * 0.1), 7)} days"
        cv_horizon = f"{horizon * (1 if unit == 'D' else 7)} days"
        print("\n" + "-" * 70)
        print(f"CROSS-VALIDATION (initial={initial}, period={period}, horizon={cv_horizon})")
        print("-" * 70)
        cv_model = make_model()
        cv_model.fit(ts)  # tüm seri: CV katları birden fazla yazı görür
        try:
            cv_df = cross_validation(
                cv_model, initial=initial, period=period, horizon=cv_horizon,
                parallel=None,
            )
            cv_df = cv_df.dropna(subset=["y", "yhat"])  # outlier maskeli noktaları hariç tut
            perf = performance_metrics(cv_df)
            # Sıfır içeren serilerde 'mape' üretilmeyebilir -> yalnızca mevcut sütunları göster
            cols = [c for c in ["horizon", "mae", "rmse", "mape", "coverage"] if c in perf.columns]
            print(perf[cols].to_string(index=False))
            print(f"\nCV ortalama MAE: {perf['mae'].mean():.3f} "
                  f"(çok-katlı; tek-yıl holdout'tan daha güvenilir)")
        except Exception as e:  # noqa: BLE001
            print(f"CV çalıştırılamadı: {e}")

    # --- Final model (tüm veri) + gelecek tahmini ---
    final_model = make_model()
    if args.with_weather:
        final_model.fit(ts)
        future = final_model.make_future_dataframe(periods=horizon, freq=args.freq)
        # Gelecekteki hava bilinmediğinden son mevsimsel ortalamayla doldur (placeholder)
        for c in WEATHER_COLS:
            month_mean = ts.assign(month=ts["ds"].dt.month).groupby("month")[c].mean()
            future_months = future["ds"].dt.month
            future[c] = ts.set_index("ds")[c].reindex(future["ds"]).values
            mask = future[c].isna()
            future.loc[mask, c] = future_months[mask].map(month_mean).values
    else:
        final_model.fit(ts)
        future = final_model.make_future_dataframe(periods=horizon, freq=args.freq)

    forecast = final_model.predict(future)
    forecast["yhat"] = np.clip(forecast["yhat"], 0, None)
    forecast["yhat_lower"] = np.clip(forecast["yhat_lower"], 0, None)
    forecast["yhat_upper"] = np.clip(forecast["yhat_upper"], 0, None)

    out_csv = os.path.join(args.out, "forecast_failures.csv")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(out_csv, index=False)

    # --- Grafikler ---
    fig1 = final_model.plot(forecast)
    plt.title(f"Filo Geneli Arıza Tahmini ({'haftalık' if args.freq == 'W' else 'günlük'})")
    plt.xlabel("Tarih")
    plt.ylabel("Arıza sayısı")
    fig1.savefig(os.path.join(args.out, "prophet_forecast.png"), dpi=120, bbox_inches="tight")
    plt.close(fig1)

    fig2 = final_model.plot_components(forecast)
    fig2.savefig(os.path.join(args.out, "prophet_components.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)

    print("\n" + "-" * 70)
    print("GELECEK TAHMİNİ (sonraki periyodlar)")
    print("-" * 70)
    tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)
    print(tail.to_string(index=False))
    print(f"\nÇıktılar:\n  {out_csv}\n  "
          f"{os.path.join(args.out, 'prophet_forecast.png')}\n  "
          f"{os.path.join(args.out, 'prophet_components.png')}")


if __name__ == "__main__":
    main()
