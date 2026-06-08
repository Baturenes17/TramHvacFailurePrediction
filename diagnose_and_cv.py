"""
Teşhis + Zaman-Serisi Cross-Validation (sınıflandırma için kararlı skor)
========================================================================
1) Her veri setinde train/test pencerelerini ve test-penceresi arıza oranını/mevsimini
   yazdırır — 2025'li vs 2025'siz ROC farkını AÇIKLAMAK için.
2) Tek bir keyfi split yerine, genişleyen-pencere TimeSeriesSplit (5 kat) ile
   ROC-AUC ve PR-AUC ortalama±std üretir — sunumda gösterilecek KARARLI skor budur.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import train_failure_model as T

warnings.filterwarnings("ignore")

DATASETS = {
    "full (2023-2025)": "3_years_data.csv",
    "no2025 (2023-2024)": "3_years_data_no2025.csv",
    "until_3871 (2023-2025)": "3_years_data_until_3871_clean.csv",
    "no2025_3871 (2023-2024)": "3_years_data_no2025_3871_clean.csv",
}


def diagnose(name, path):
    df = T.engineer_features(T.load_data(path))
    train, test = T.time_based_split(df)
    tr_pos = train[T.TARGET].mean(); te_pos = test[T.TARGET].mean()
    # test penceresindeki ay dağılımı (yaz = yüksek arıza sezonu mu?)
    test_months = test["date"].dt.month
    summer_frac = test_months.isin([6, 7, 8]).mean()
    print(f"\n{name}")
    print(f"  TRAIN: {train['date'].min().date()} -> {train['date'].max().date()} "
          f"(arıza oranı {tr_pos:.3f})")
    print(f"  TEST : {test['date'].min().date()} -> {test['date'].max().date()} "
          f"(arıza oranı {te_pos:.3f}, yaz-ayı payı {summer_frac:.2f})")


def cv_score(name, path, model_type="lightgbm", n_splits=5):
    df = T.engineer_features(T.load_data(path)).sort_values("date").reset_index(drop=True)
    feat = T.get_feature_columns(df)
    y = df[T.TARGET].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rocs, aps, precs, recs = [], [], [], []
    for tr_idx, te_idx in tscv.split(df):
        prep = T.build_preprocessor(feat)
        Xtr = prep.fit_transform(df.iloc[tr_idx][feat]); Xte = prep.transform(df.iloc[te_idx][feat])
        spw = T.compute_scale_pos_weight(pd.Series(y[tr_idx]))
        m = T.build_model(model_type, spw); m.fit(Xtr, y[tr_idx])
        s = m.predict_proba(Xte)[:, 1]
        yte = y[te_idx]
        if yte.sum() == 0:
            continue
        rocs.append(roc_auc_score(yte, s)); aps.append(average_precision_score(yte, s))
        thr = np.quantile(s, 0.9); pred = (s >= thr).astype(int)
        precs.append(precision_score(yte, pred, zero_division=0))
        recs.append(recall_score(yte, pred, zero_division=0))
    print(f"  {name:<26}{model_type:<10} "
          f"ROC {np.mean(rocs):.3f}±{np.std(rocs):.3f} | "
          f"PR-AUC {np.mean(aps):.3f}±{np.std(aps):.3f} | "
          f"P@10% {np.mean(precs):.3f} | R@10% {np.mean(recs):.3f} "
          f"(base {y.mean():.3f})")


def main():
    print("=" * 78)
    print("1) TEŞHİS — train/test pencereleri ve test-penceresi mevsimi")
    print("=" * 78)
    for name, path in DATASETS.items():
        diagnose(name, path)

    print("\n" + "=" * 78)
    print("2) ZAMAN-SERİSİ CV (genişleyen pencere, 5 kat) — KARARLI sınıflandırma skoru")
    print("=" * 78)
    for model in ["lightgbm", "logreg"]:
        print(f"\n--- model: {model} ---")
        for name, path in DATASETS.items():
            cv_score(name, path, model)


if __name__ == "__main__":
    main()
