# -*- coding: utf-8 -*-
"""
Komut çıktıları için grafik üretici.
=====================================
1-3) train_failure_model.py LightGBM koşuları varsayılan olarak grafik üretmez
     (sadece CSV/metin). Burada rapor için anlamlı DEĞERLENDİRME grafiklerini
     üretiriz: ROC eğrisi (train+test), Precision-Recall eğrisi (test),
     Confusion matrix (test, fixed eşik). Ana pipeline fonksiyonları yeniden
     kullanılır -> komut çıktısıyla bire bir aynı model/eşik.
5)   Prophet --cv için cross-validation MAE-vs-horizon grafiği.

Prophet'in kendi forecast/components grafikleri (komut 4 ve 5) doğrudan
forecast_failures_prophet.py --out ile üretilir (bkz. çalıştırma komutları).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

from train_failure_model import (
    TARGET, CATEGORICAL_FEATURES,
    load_data, engineer_features, get_feature_columns, time_based_split,
    build_preprocessor, build_model, compute_scale_pos_weight,
    recall_constrained_threshold,
)

OUT = "outputs/grafikler"
os.makedirs(OUT, exist_ok=True)
TEST_END = pd.to_datetime("2025-12-01")


def run_classification(data, smote, cmd_label, fname):
    """Komutun pipeline'ını birebir tekrarla, 3 panelli değerlendirme grafiği üret."""
    df = engineer_features(load_data(data))
    feature_cols = get_feature_columns(df)
    train, test = time_based_split(df, test_end=TEST_END)

    prep = build_preprocessor(feature_cols)
    Xtr = prep.fit_transform(train[feature_cols]); ytr = train[TARGET].values
    Xte = prep.transform(test[feature_cols]);      yte = test[TARGET].values

    if smote:
        from imblearn.over_sampling import SMOTENC
        n_num = len([c for c in feature_cols if c not in CATEGORICAL_FEATURES])
        n_cat = len([c for c in feature_cols if c in CATEGORICAL_FEATURES])
        cat_idx = list(range(n_num, n_num + n_cat))
        sm = SMOTENC(categorical_features=cat_idx, sampling_strategy="auto", random_state=42)
        Xtr, ytr = sm.fit_resample(Xtr, ytr)

    spw = compute_scale_pos_weight(pd.Series(ytr))
    model = build_model("lightgbm", spw, None)
    model.fit(Xtr, ytr)

    tr_scores = model.predict_proba(Xtr)[:, 1]
    te_scores = model.predict_proba(Xte)[:, 1]

    # fixed eşik = train'de recall>=0.80 altında precision-maks (komutla aynı)
    op = recall_constrained_threshold(ytr, tr_scores, 0.80)
    thr = op[0] if op else 0.5
    yte_pred = (te_scores >= thr).astype(int)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- Panel 1: ROC eğrisi (train + test) ---
    for scores, y, name in [(tr_scores, ytr, "Train"), (te_scores, yte, "Test")]:
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Eğrisi")
    ax[0].legend(loc="lower right")

    # --- Panel 2: Precision-Recall eğrisi (test) ---
    prec, rec, _ = precision_recall_curve(yte, te_scores)
    ap = average_precision_score(yte, te_scores)
    ax[1].plot(rec, prec, color="C1", label=f"Test PR-AUC (AP)={ap:.3f}")
    base = yte.mean()
    ax[1].axhline(base, ls="--", color="gray", alpha=0.6, label=f"Baz oran={base:.3f}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Eğrisi (Test)")
    ax[1].legend(loc="upper right")

    # --- Panel 3: Confusion matrix (test, fixed eşik) ---
    cm = confusion_matrix(yte, yte_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Normal(0)", "Arıza(1)"]).plot(
        ax=ax[2], cmap="Blues", colorbar=False)
    ax[2].set_title(f"Confusion Matrix (Test, eşik={thr:.3f})")

    te_auc = roc_auc_score(yte, te_scores)
    fig.suptitle(f"{cmd_label}  |  Test ROC-AUC={te_auc:.4f}  PR-AUC={ap:.4f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def prophet_cv_plot():
    """Komut 5 (--cv) için cross-validation MAE-vs-horizon grafiği."""
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    from forecast_failures_prophet import (
        load_data as p_load, derive_failure_events, build_timeseries,
    )

    df = derive_failure_events(p_load("3_years_data_no2025.csv"))
    ts = build_timeseries(df, freq="W", with_weather=False)
    span = (ts["ds"].max() - ts["ds"].min()).days
    initial = f"{int(span * 0.5)} days"
    period = f"{max(int(span * 0.1), 7)} days"
    horizon = f"{8 * 7} days"

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode="additive", interval_width=0.80)
    m.fit(ts)
    cv_df = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel=None)
    cv_df = cv_df.dropna(subset=["y", "yhat"])
    perf = performance_metrics(cv_df)
    perf["horizon_days"] = perf["horizon"].dt.days

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(perf["horizon_days"], perf["mae"], marker="o", color="C2", label="CV MAE")
    ax.axhline(perf["mae"].mean(), ls="--", color="gray",
               label=f"Ortalama MAE={perf['mae'].mean():.3f}")
    ax.set(xlabel="Tahmin ufku (gün)", ylabel="MAE",
           title="Komut 5 — Prophet Cross-Validation (MAE vs Ufuk)\n3_years_data_no2025.csv")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT, "cmd5_prophet_cv_mae.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


if __name__ == "__main__":
    run_classification("3_years_data_no7d.csv", False,
                       "Komut 1 — no7d (dengeleme yok)", "cmd1_no7d.png")
    run_classification("3_years_data_no7d_undersampled.csv", False,
                       "Komut 2 — no7d_undersampled", "cmd2_undersampled.png")
    run_classification("3_years_data_no7d_undersampled.csv", True,
                       "Komut 3 — no7d_undersampled + SMOTE", "cmd3_undersampled_smote.png")
    prophet_cv_plot()
    print("\nTüm sınıflandırma grafikleri + CV grafiği üretildi.")
