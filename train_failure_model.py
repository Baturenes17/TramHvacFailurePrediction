"""
Tram HVAC Arıza Tahmini — Eğitim & Tahmin Pipeline'ı
====================================================

Amaç: `3_years_data.csv` içindeki `failure_next_30d` etiketini kullanarak bir aracın
önümüzdeki 30 gün içinde HVAC arızası yapıp yapmayacağını tahmin etmek.

Akış:
  1. Veriyi yükle (`;` ayraçlı, `,` ondalık).
  2. Sızıntısız (causal) feature engineering uygula.
  3. Takvim-bazlı 60/20/20 split (gelecekten geçmişe sızıntı yok).
  4. LightGBM eğit (sınıf dengesizliği için scale_pos_weight + early stopping).
  5. Validation + test üzerinde ROC-AUC / PR-AUC / classification_report raporla.
  6. Eşik seç (alarm-oranı veya sabit eşik).
  7. (Opsiyonel) Optuna ile hiperparametre araması, SHAP ile özellik önemi.
  8. Tüm veriyle final modeli fit edip her aracın EN GÜNCEL gününü skorla ->
     `outputs/predictions_latest.csv` (riske göre sıralı araç listesi).

Ayrıntılı dokümantasyon için bkz. PROJE.md
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# 1. Konfigürasyon
# --------------------------------------------------------------------------- #
PREDICTION_HORIZON_DAYS = 30          # Bu sürümde sabit: failure_next_30d
DEFAULT_DATA = "3_years_data.csv"
DEFAULT_OUT = "outputs"
ALARM_RATE = 0.40                     # Her dönemde en riskli %40 araç pozitif işaretlenir
RANDOM_STATE = 42
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20     # Test = kalan %20 (takvim bazlı)
EARLY_STOPPING_ROUNDS = 50

TARGET = f"failure_next_{PREDICTION_HORIZON_DAYS}d"
OTHER_LABEL = "failure_next_7d"       # Sızıntıyı önlemek için özelliklerden çıkarılır
ID_COLS = ["date", "vehicle_id"]
RECALL_TARGETS = [0.90, 0.80, 0.70, 0.60, 0.50]

# Kategorik olarak ele alınacak ham/üretilmiş sütunlar
CATEGORICAL_FEATURES = ["vehicle_type", "weather_type", "season"]


# --------------------------------------------------------------------------- #
# 2. Veri yükleme
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["vehicle_id", "date"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 3. Feature engineering (zaman-bilinçli, sızıntısız)
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm türetilen özellikler yalnızca geçmiş/aynı-gün bilgisinden gelir.
    Araç-bazlı yuvarlanan istatistikler shift(1) ile kaydırılır (gelecek sızıntısı yok).
    """
    df = df.copy()
    eps = 1e-6

    # --- Takvim / mevsim ---
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["season"] = (df["month"] % 12 // 3).map(
        {0: "winter", 1: "spring", 2: "summer", 3: "autumn"}
    )
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

    # --- Hava-stres ---
    df["temp_sq"] = df["temp"] ** 2
    df["is_hot"] = (df["temp"] >= 30).astype(int)
    df["is_cold"] = (df["temp"] <= 0).astype(int)
    df["temp_x_humidity"] = df["temp"] * df["humidity"]
    # Aylık iklim ortalamasından sapma (exogen hava değişkeni — etiketten türetilmez)
    month_mean_temp = df.groupby("month")["temp"].transform("mean")
    df["temp_dev_from_month"] = df["temp"] - month_mean_temp

    # --- Kullanım trendi (araç bazında, causal) ---
    df["km_7d_30d_ratio"] = df["km_last_7d"] / (df["km_last_30d"] + eps)
    g = df.groupby("vehicle_id")["km_today"]
    df["km_roll30_std"] = g.transform(
        lambda s: s.shift(1).rolling(30, min_periods=5).std()
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Modelin kullanacağı özellik sütunları: kimlik, hedef ve sızıntı kolonları hariç."""
    exclude = set(ID_COLS) | {TARGET, OTHER_LABEL}
    return [c for c in df.columns if c not in exclude]


# --------------------------------------------------------------------------- #
# 4. Takvim bazlı bölme
# --------------------------------------------------------------------------- #
def time_based_split(df: pd.DataFrame):
    """Tarih aralığına göre 60/20/20 (takvim bazlı). Kronolojik kesim — sızıntı yok."""
    df_sorted = df.sort_values("date").reset_index(drop=True)
    dmin, dmax = df_sorted["date"].min(), df_sorted["date"].max()
    span = dmax - dmin
    train_end = dmin + span * TRAIN_FRAC
    val_end = dmin + span * (TRAIN_FRAC + VAL_FRAC)
    train = df_sorted[df_sorted["date"] <= train_end]
    val = df_sorted[(df_sorted["date"] > train_end) & (df_sorted["date"] <= val_end)]
    test = df_sorted[df_sorted["date"] > val_end]
    return train, val, test


def _fmt_range(s: pd.DataFrame) -> str:
    return f"{s['date'].min()} - {s['date'].max()}"


# --------------------------------------------------------------------------- #
# 5. Önişleme + Model
# --------------------------------------------------------------------------- #
def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    cat = [c for c in feature_cols if c in CATEGORICAL_FEATURES]
    num = [c for c in feature_cols if c not in cat]

    numeric_tf = SimpleImputer(strategy="median")
    categorical_tf = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num),
            ("cat", categorical_tf, cat),
        ],
        remainder="drop",
    )


def build_model(scale_pos_weight: float, params: dict | None = None) -> LGBMClassifier:
    """LightGBM sınıflandırıcı. Bu sürümde tek model; ileride model adına göre
    dallanma eklemek için ayrı fonksiyon olarak tutuldu."""
    base = dict(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        base.update(params)
    return LGBMClassifier(**base)


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = max(int(y.sum()), 1)
    neg = int((y == 0).sum())
    return neg / pos


# --------------------------------------------------------------------------- #
# 6. Eşik yardımcıları
# --------------------------------------------------------------------------- #
def alarm_rate_threshold(scores: np.ndarray, alarm_rate: float) -> float:
    """Skorların en yüksek `alarm_rate` oranı pozitif olacak şekilde eşik."""
    return float(np.quantile(scores, 1.0 - alarm_rate))


def recall_constrained_threshold(y_true, scores, target_recall: float):
    """recall >= target altında precision-maks eşik (en yüksek uygun eşik). (eşik, precision)."""
    ths = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 400)))
    best = None
    for t in ths:
        pred = (scores >= t).astype(int)
        if recall_score(y_true, pred, zero_division=0) >= target_recall:
            best = (float(t), precision_score(y_true, pred, zero_division=0))
    return best


def f1_optimal_threshold(y_true, scores) -> float:
    ths = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 400)))
    best_t, best_f = 0.5, -1.0
    for t in ths:
        f = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


# --------------------------------------------------------------------------- #
# 7. Raporlama
# --------------------------------------------------------------------------- #
def report_split(title: str, y_true, scores, threshold: float) -> float:
    pred = (scores >= threshold).astype(int)
    roc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    rate = float(np.mean(pred))
    print(f"\n==== {title} ROC-AUC: {roc:.4f} | PR-AUC (AP): {ap:.4f} ====")
    print(f"[eşik={threshold:.3f} | işaretlenen oran={rate:.3f}]")
    print(classification_report(y_true, pred, digits=4))
    return roc


def precision_recall_tradeoff(y_true, scores):
    print("Precision/Recall ödünleşimi (arıza sınıfı):")
    print("  recall>=  precision   eşik")
    for tg in RECALL_TARGETS:
        res = recall_constrained_threshold(y_true, scores, tg)
        if res:
            t, p = res
            print(f"   {tg:.2f}       {p:.3f}     {t:.3f}")
        else:
            print(f"   {tg:.2f}         -         -")


# --------------------------------------------------------------------------- #
# 8. Optuna ile hiperparametre araması (opsiyonel)
# --------------------------------------------------------------------------- #
def tune_optuna(df_trainval, feature_cols, n_trials):
    import optuna
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df_sorted = df_trainval.sort_values("date").reset_index(drop=True)
    X = df_sorted[feature_cols]
    y = df_sorted[TARGET].values
    spw = compute_scale_pos_weight(df_sorted[TARGET])
    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        scores = []
        prep = build_preprocessor(feature_cols)
        for tr_idx, va_idx in tscv.split(X):
            Xtr = prep.fit_transform(X.iloc[tr_idx])
            Xva = prep.transform(X.iloc[va_idx])
            model = build_model(spw, params)
            model.fit(Xtr, y[tr_idx])
            p = model.predict_proba(Xva)[:, 1]
            scores.append(average_precision_score(y[va_idx], p))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"\n[Optuna] En iyi PR-AUC (CV): {study.best_value:.4f}")
    print(f"[Optuna] En iyi parametreler: {study.best_params}")
    return study.best_params


# --------------------------------------------------------------------------- #
# 9. SHAP (opsiyonel)
# --------------------------------------------------------------------------- #
def run_shap(prep, model, X_sample, out_dir):
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_trans = prep.transform(X_sample)
    feature_names = prep.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    path = os.path.join(out_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[SHAP] Özet plot kaydedildi -> {path}")


# --------------------------------------------------------------------------- #
# 10. Tahmin çıktısı
# --------------------------------------------------------------------------- #
def score_latest(full_df, prep, model, feature_cols, alarm_rate, out_dir):
    """Her aracın en güncel gününü skorla, riske göre sırala, CSV yaz."""
    latest = (
        full_df.sort_values("date")
        .groupby("vehicle_id", as_index=False)
        .tail(1)
        .copy()
    )
    scores = model.predict_proba(prep.transform(latest[feature_cols]))[:, 1]
    latest["failure_prob_30d"] = scores
    latest = latest.sort_values("failure_prob_30d", ascending=False).reset_index(drop=True)
    latest["risk_rank"] = np.arange(1, len(latest) + 1)
    thr = alarm_rate_threshold(scores, alarm_rate)
    latest["alarm_flag"] = (latest["failure_prob_30d"] >= thr).astype(int)

    cols = ["risk_rank", "vehicle_id", "vehicle_type", "date",
            "failure_prob_30d", "alarm_flag"]
    out = latest[cols]
    path = os.path.join(out_dir, "predictions_latest.csv")
    out.to_csv(path, sep=";", decimal=",", index=False)
    print(f"\n[Tahmin] {len(out)} aracın güncel risk skoru -> {path}")
    print(f"[Tahmin] Alarm eşiği (en riskli %{alarm_rate*100:.0f}): {thr:.4f}, "
          f"alarm sayısı: {int(out['alarm_flag'].sum())}")
    print("\nEn riskli 10 araç:")
    with pd.option_context("display.max_rows", 10, "display.width", 120):
        print(out.head(10).to_string(index=False))
    return out


def predict_file(path, prep, model, feature_cols, alarm_rate, out_dir):
    """Yeni bir CSV'yi (gelecekte toplanan veri) aynı pipeline'dan geçirip skorlar."""
    df = engineer_features(load_data(path))
    scores = model.predict_proba(prep.transform(df[feature_cols]))[:, 1]
    df["failure_prob_30d"] = scores
    thr = alarm_rate_threshold(scores, alarm_rate)
    df["alarm_flag"] = (df["failure_prob_30d"] >= thr).astype(int)
    out_path = os.path.join(out_dir, "predictions_custom.csv")
    keep = ["date", "vehicle_id", "vehicle_type", "failure_prob_30d", "alarm_flag"]
    df.sort_values("failure_prob_30d", ascending=False)[keep].to_csv(
        out_path, sep=";", decimal=",", index=False
    )
    print(f"\n[--predict] {len(df)} kayıt skorlandı -> {out_path}")


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Tram HVAC 30 günlük arıza tahmini")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Eğitim CSV yolu")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Çıktı klasörü")
    parser.add_argument("--tune", action="store_true", help="Optuna hiperparametre araması")
    parser.add_argument("--trials", type=int, default=40, help="Optuna deneme sayısı")
    parser.add_argument("--shap", action="store_true", help="SHAP özellik önemi üret")
    parser.add_argument("--alarm-rate", type=float, default=ALARM_RATE,
                        help="Alarm-oranı eşiği için pozitif yüzdesi (0-1)")
    parser.add_argument("--threshold-mode", choices=["alarm", "fixed"], default="alarm",
                        help="Eşik seçim modu")
    parser.add_argument("--predict", default=None,
                        help="Eğitim sonrası skorlanacak ek CSV yolu")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) Yükle
    df_raw = load_data(args.data)
    print("Data shape:", df_raw.shape)
    print("First rows:")
    print(df_raw.head())

    # 2-3) Feature engineering + sızıntı kontrolü
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns(df)
    assert TARGET not in feature_cols and OTHER_LABEL not in feature_cols, "Sızıntı!"

    # 4) Takvim bazlı bölme
    train, val, test = time_based_split(df)
    print(f"Train size: {len(train)}  dates: {_fmt_range(train)}")
    print(f"Val   size: {len(val)}  dates: {_fmt_range(val)}")
    print(f"Test  size: {len(test)}  dates: {_fmt_range(test)}")

    # (Opsiyonel) Optuna
    best_params = None
    if args.tune:
        best_params = tune_optuna(pd.concat([train, val]), feature_cols, args.trials)

    # 5) Eğitim — preprocessor + LightGBM (early stopping val üzerinde)
    spw = compute_scale_pos_weight(train[TARGET])
    prep = build_preprocessor(feature_cols)
    Xtr = prep.fit_transform(train[feature_cols])
    Xva = prep.transform(val[feature_cols])
    Xte = prep.transform(test[feature_cols])
    ytr, yva, yte = train[TARGET].values, val[TARGET].values, test[TARGET].values

    model = build_model(spw, best_params)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
    )
    best_iter = model.best_iteration_ or model.n_estimators
    print(f"[LightGBM] best iteration (early stopping): {best_iter}")

    # 6) Skorlar + eşikler
    val_scores = model.predict_proba(Xva)[:, 1]
    test_scores = model.predict_proba(Xte)[:, 1]

    op = recall_constrained_threshold(yva, val_scores, 0.80)
    op_t = op[0] if op else float("nan")
    f1opt = f1_optimal_threshold(yva, val_scores)

    if args.threshold_mode == "alarm":
        val_thr = alarm_rate_threshold(val_scores, args.alarm_rate)
        test_thr = alarm_rate_threshold(test_scores, args.alarm_rate)
        print(f"Eşik modu — alarm-oranı (her dönemde en riskli %{args.alarm_rate*100:.0f}) "
              f"| referans: op(recall>=0.80)={op_t:.3f}, F1-opt={f1opt:.3f}")
    else:
        val_thr = test_thr = op_t
        print(f"Eşik modu — sabit eşik (val recall>=0.80)={op_t:.3f} "
              f"| referans: F1-opt={f1opt:.3f}")

    # 7) Per-split raporlar
    print("\n--- VALIDATION ---")
    val_roc = report_split("Validation", yva, val_scores, val_thr)
    precision_recall_tradeoff(yva, val_scores)

    print("\n--- TEST ---")
    test_roc = report_split("Test", yte, test_scores, test_thr)
    precision_recall_tradeoff(yte, test_scores)

    # 9) SHAP
    if args.shap:
        sample = test[feature_cols].sample(min(2000, len(test)), random_state=RANDOM_STATE)
        run_shap(prep, model, sample, args.out)

    print("\nModel training completed.")
    print("Validation ROC-AUC:", val_roc)
    print("Test ROC-AUC:", test_roc)

    # 10) Final model: TÜM veriyle yeniden fit -> güncel skorlama
    spw_full = compute_scale_pos_weight(df[TARGET])
    final_params = {**(best_params or {}), "n_estimators": int(best_iter)}
    final_prep = build_preprocessor(feature_cols)
    Xall = final_prep.fit_transform(df[feature_cols])
    final_model = build_model(spw_full, final_params)
    final_model.fit(Xall, df[TARGET].values)

    score_latest(df, final_prep, final_model, feature_cols, args.alarm_rate, args.out)

    if args.predict:
        predict_file(args.predict, final_prep, final_model, feature_cols,
                     args.alarm_rate, args.out)


if __name__ == "__main__":
    main()
