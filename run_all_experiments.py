"""
TÜM DENEYLER — İkili Sınıflandırma vs Zaman Serisi (Prophet)
============================================================
Bitirme projesi için dürüst, sızıntısız karşılaştırma. 4 veri setinin her birinde:

  A) İKİLİ SINIFLANDIRMA (failure_next_30d): train_failure_model'in AYNI sızıntısız
     feature engineering + zaman-bazlı 80/20 split'i. Modeller: LightGBM, Regularized
     LightGBM, XGBoost, CatBoost, RandomForest, LogReg, Ensemble.
     Test metrikleri: ROC-AUC, PR-AUC, en riskli %10'da precision/recall,
     ayrıca recall>=0.80'de precision.

  B) ZAMAN SERİSİ (Prophet): filo geneli haftalık arıza sayısı. Kronolojik holdout'ta
     Prophet vs Naive vs Mevsimsel-naive (MAE/RMSE) + cross-validation ortalama MAE.

Çıktı: outputs/experiment_results.csv (sınıflandırma) ve ekrana özet tablo.
"""
from __future__ import annotations

import logging
import warnings
import numpy as np
import pandas as pd

# Prophet/cmdstanpy gürültüsünü bastır
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
)
from sklearn.ensemble import RandomForestClassifier

import train_failure_model as T
import forecast_failures_prophet as P

warnings.filterwarnings("ignore")

DATASETS = [
    "3_years_data.csv",
    "3_years_data_no2025.csv",
    "3_years_data_until_3871_clean.csv",
    "3_years_data_no2025_3871_clean.csv",
]
ALARM_RATE = 0.10


# --------------------------------------------------------------------------- #
# A) İKİLİ SINIFLANDIRMA
# --------------------------------------------------------------------------- #
def eval_classification(dataset: str) -> list[dict]:
    df = T.engineer_features(T.load_data(dataset))
    feat = T.get_feature_columns(df)
    train, test = T.time_based_split(df)

    prep = T.build_preprocessor(feat)
    Xtr = prep.fit_transform(train[feat]); ytr = train[T.TARGET].values
    Xte = prep.transform(test[feat]);       yte = test[T.TARGET].values
    spw = T.compute_scale_pos_weight(train[T.TARGET])

    base_rate = float(yte.mean())  # rastgele tahminin precision'ı bu olur

    def metrics(name, scores):
        thr = float(np.quantile(scores, 1 - ALARM_RATE))
        pred = (scores >= thr).astype(int)
        # recall>=0.80 altında ulaşılabilir en iyi precision (train eşiğiyle, sızıntısız)
        op = T.recall_constrained_threshold(ytr, scores_train_map[name], 0.80) \
            if name in scores_train_map else None
        prec_at_r80 = op[1] if op else float("nan")
        return dict(
            dataset=dataset, model=name,
            roc=roc_auc_score(yte, scores),
            pr_auc=average_precision_score(yte, scores),
            base_rate=base_rate,
            prec_top10=precision_score(yte, pred, zero_division=0),
            rec_top10=recall_score(yte, pred, zero_division=0),
            lift_top10=precision_score(yte, pred, zero_division=0) / base_rate,
            prec_at_recall80=prec_at_r80,
        )

    scores_train_map: dict[str, np.ndarray] = {}
    results = []

    def add(name, model):
        model.fit(Xtr, ytr)
        scores_train_map[name] = model.predict_proba(Xtr)[:, 1]
        results.append(metrics(name, model.predict_proba(Xte)[:, 1]))

    add("LightGBM", T.build_model("lightgbm", spw))
    add("LightGBM-Reg", T.build_model("lightgbm", spw, T.REGULARIZED_LGBM_PARAMS))
    add("XGBoost", T.build_model("xgboost", spw))
    add("CatBoost", T.build_model("catboost", spw))
    add("RandomForest", RandomForestClassifier(
        n_estimators=400, min_samples_leaf=20, class_weight="balanced",
        n_jobs=-1, random_state=T.RANDOM_STATE))
    add("LogReg", T.build_model("logreg", spw))
    add("Ensemble", T.build_model("ensemble", spw))

    return results


# --------------------------------------------------------------------------- #
# B) ZAMAN SERİSİ (Prophet) — holdout + CV
# --------------------------------------------------------------------------- #
def eval_prophet(dataset: str, freq: str = "W") -> dict:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics

    df = P.load_data(dataset)
    df = P.derive_failure_events(df)
    ts = P.build_timeseries(df, freq=freq, with_weather=False)
    train, holdout = P.time_based_split(ts)

    def make():
        return Prophet(yearly_seasonality=True, weekly_seasonality=(freq == "D"),
                       daily_seasonality=False, seasonality_mode="additive",
                       interval_width=0.80)

    m = make(); m.fit(train)
    fc = m.predict(holdout[["ds"]])
    yhat = np.clip(fc["yhat"].values, 0, None)
    yt = holdout["y"].values
    valid = ~np.isnan(yt)
    mp = P._metrics(yt[valid], yhat[valid])
    naive = np.full(len(holdout), np.nanmean(train["y"]))
    mn = P._metrics(yt[valid], naive[valid])
    snaive = P.seasonal_naive_pred(ts, holdout, freq)
    ms = P._metrics(yt[valid], snaive[valid])

    # CV ortalama MAE (çok-katlı; daha güvenilir)
    cv_mae = float("nan")
    try:
        span = (ts["ds"].max() - ts["ds"].min()).days
        cvm = make(); cvm.fit(ts)
        cv_df = cross_validation(
            cvm, initial=f"{int(span*0.5)} days",
            period=f"{max(int(span*0.1),7)} days",
            horizon=f"{8*7} days", parallel=None)
        cv_df = cv_df.dropna(subset=["y", "yhat"])
        cv_mae = float(performance_metrics(cv_df)["mae"].mean())
    except Exception as e:
        print(f"  [Prophet CV hata] {dataset}: {e}")

    return dict(
        dataset=dataset, freq=freq, n_periods=len(ts),
        mean_y=float(ts["y"].mean()),
        prophet_mae=mp["MAE"], prophet_rmse=mp["RMSE"],
        naive_mae=mn["MAE"], snaive_mae=ms["MAE"],
        cv_mae=cv_mae,
        prophet_beats_naive=mp["MAE"] < mn["MAE"],
        prophet_beats_snaive=mp["MAE"] < ms["MAE"],
    )


# --------------------------------------------------------------------------- #
def main():
    print("#" * 78)
    print("# A) İKİLİ SINIFLANDIRMA — failure_next_30d (test seti, zaman-bazlı split)")
    print("#" * 78)
    cls_rows = []
    for ds in DATASETS:
        print(f"\n>>> {ds}")
        rows = eval_classification(ds)
        cls_rows.extend(rows)
        print(f"{'Model':<14}{'ROC':>7}{'PR-AUC':>8}{'base':>7}"
              f"{'P@10%':>8}{'R@10%':>8}{'Lift':>7}{'P@R80':>8}")
        for r in sorted(rows, key=lambda x: x["pr_auc"], reverse=True):
            print(f"{r['model']:<14}{r['roc']:>7.3f}{r['pr_auc']:>8.3f}"
                  f"{r['base_rate']:>7.3f}{r['prec_top10']:>8.3f}"
                  f"{r['rec_top10']:>8.3f}{r['lift_top10']:>7.2f}"
                  f"{r['prec_at_recall80']:>8.3f}")

    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv("outputs/experiment_results.csv", index=False)

    print("\n\n" + "#" * 78)
    print("# B) ZAMAN SERİSİ — Prophet (filo geneli haftalık arıza sayısı)")
    print("#" * 78)
    ts_rows = []
    for ds in DATASETS:
        print(f"\n>>> {ds}")
        r = eval_prophet(ds, freq="W")
        ts_rows.append(r)
        print(f"  Holdout MAE — Prophet:{r['prophet_mae']:.3f}  "
              f"Naive:{r['naive_mae']:.3f}  Mevsimsel-naive:{r['snaive_mae']:.3f}")
        print(f"  Prophet RMSE:{r['prophet_rmse']:.3f}  CV-ort MAE:{r['cv_mae']:.3f}  "
              f"(ort y={r['mean_y']:.2f}/hafta)")
        verdict = "Prophet KAZANDI" if (r['prophet_beats_naive'] and r['prophet_beats_snaive']) \
            else ("baseline'ları geçemedi" if not r['prophet_beats_naive'] else "naive'i geçti, mevsimseli geçemedi")
        print(f"  -> {verdict}")

    pd.DataFrame(ts_rows).to_csv("outputs/prophet_results.csv", index=False)

    # ---- ÖZET ----
    print("\n\n" + "=" * 78)
    print("ÖZET — En iyi sınıflandırma (PR-AUC) ve zaman serisi sonuçları")
    print("=" * 78)
    best = cls_df.loc[cls_df.groupby("dataset")["pr_auc"].idxmax()]
    print("\nHer veri setinde en iyi sınıflandırıcı (PR-AUC'a göre):")
    print(best[["dataset", "model", "roc", "pr_auc", "prec_top10",
                "rec_top10", "lift_top10"]].to_string(index=False))


if __name__ == "__main__":
    main()
