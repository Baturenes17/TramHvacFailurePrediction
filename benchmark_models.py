"""
Model karşılaştırması — precision'ı yükseltmek farklı modelle mümkün mü?
=======================================================================
train_failure_model.py'deki AYNI feature engineering + AYNI zaman-bazlı
split kullanılır. Tek değişen: sınıflandırıcı. Her model için test setinde
PR-AUC, ROC-AUC ve sabit çalışma noktasında (en riskli %10) precision/recall.
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_failure_model as T

warnings.filterwarnings("ignore")

ALARM_RATE = 0.10  # sabit çalışma noktası: en riskli %10


def main():
    df = T.engineer_features(T.load_data(T.DEFAULT_DATA))
    feat = T.get_feature_columns(df)
    train, test = T.time_based_split(df)

    prep = T.build_preprocessor(feat)
    Xtr = prep.fit_transform(train[feat]); ytr = train[T.TARGET].values
    Xte = prep.transform(test[feat]);       yte = test[T.TARGET].values
    spw = T.compute_scale_pos_weight(train[T.TARGET])

    def evaluate(name, scores):
        thr = float(np.quantile(scores, 1 - ALARM_RATE))
        pred = (scores >= thr).astype(int)
        return dict(
            model=name,
            roc=roc_auc_score(yte, scores),
            ap=average_precision_score(yte, scores),
            prec=precision_score(yte, pred, zero_division=0),
            rec=recall_score(yte, pred, zero_division=0),
        )

    results = []

    # 1) LightGBM (mevcut baseline)
    m = T.build_model("lightgbm", spw); m.fit(Xtr, ytr)
    results.append(evaluate("LightGBM (mevcut)", m.predict_proba(Xte)[:, 1]))

    # 2) XGBoost
    from xgboost import XGBClassifier
    m = XGBClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
        eval_metric="logloss", n_jobs=-1, random_state=T.RANDOM_STATE,
    )
    m.fit(Xtr, ytr)
    results.append(evaluate("XGBoost", m.predict_proba(Xte)[:, 1]))

    # 3) CatBoost
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(
        iterations=300, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        scale_pos_weight=spw, random_seed=T.RANDOM_STATE, verbose=0,
    )
    m.fit(Xtr, ytr)
    results.append(evaluate("CatBoost", m.predict_proba(Xte)[:, 1]))

    # 4) RandomForest
    m = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=T.RANDOM_STATE,
    )
    m.fit(Xtr, ytr)
    results.append(evaluate("RandomForest", m.predict_proba(Xte)[:, 1]))

    # 5) HistGradientBoosting (sklearn)
    m = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_depth=4,
        class_weight="balanced", random_state=T.RANDOM_STATE,
    )
    m.fit(Xtr, ytr)
    results.append(evaluate("HistGradBoost", m.predict_proba(Xte)[:, 1]))

    # 6) Logistic Regression (ölçeklenmiş, lineer baseline)
    m = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    m.fit(Xtr, ytr)
    results.append(evaluate("LogisticReg", m.predict_proba(Xte)[:, 1]))

    print("\n=== TEST seti — en riskli %10 çalışma noktası ===")
    print(f"{'Model':<20}{'ROC-AUC':>9}{'PR-AUC':>9}{'Precision':>11}{'Recall':>9}")
    print("-" * 58)
    for r in sorted(results, key=lambda x: x["prec"], reverse=True):
        print(f"{r['model']:<20}{r['roc']:>9.4f}{r['ap']:>9.4f}"
              f"{r['prec']:>11.4f}{r['rec']:>9.4f}")


if __name__ == "__main__":
    main()
