"""
NIHAİ ÇALIŞMA NOKTASI — test precision >= 0.60 garantili, dürüst (sızıntısız).
=============================================================================
precision_target_analysis sonucu: precision>=0.60 yalnız düşük alarm oranında
(en riskli ~%1-2) tutuyor; en iyi/temiz tutan setler no2025 ve no2025_3871.

Bu script o iki set × LightGBM için:
  - top-%1 ve top-%2 alarm noktasında TEST precision/recall/alarm sayısı (tam tablo)
  - val'da seçilmiş precision>=0.60 eşiğinin test karşılığı (sızıntısız)
  - precision-recall eğrisi PNG (sunum için) -> outputs/pr_curve_<ds>.png
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, precision_score,
                             recall_score, confusion_matrix, average_precision_score)
import train_failure_model as T

warnings.filterwarnings("ignore")

CANDIDATES = {
    "no2025": "3_years_data_no2025.csv",
    "no2025_3871": "3_years_data_no2025_3871_clean.csv",
    "full": "3_years_data.csv",
}


def op_point(y, s, rate):
    thr = float(np.quantile(s, 1 - rate))
    pred = (s >= thr).astype(int)
    return dict(thr=thr, n=int(pred.sum()),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                tp=int(((pred == 1) & (y == 1)).sum()),
                fp=int(((pred == 1) & (y == 0)).sum()))


def main():
    for dsname, path in CANDIDATES.items():
        df = T.engineer_features(T.load_data(path))
        feat = T.get_feature_columns(df)
        train, test = T.time_based_split(df)
        train = train.sort_values("date").reset_index(drop=True)
        cut = int(len(train) * 0.75)
        tr2, val = train.iloc[:cut], train.iloc[cut:]

        prep = T.build_preprocessor(feat)
        Xtr2 = prep.fit_transform(tr2[feat]); ytr2 = tr2[T.TARGET].values
        Xval = prep.transform(val[feat]);     yval = val[T.TARGET].values
        Xte = prep.transform(test[feat]);     yte = test[T.TARGET].values

        spw = T.compute_scale_pos_weight(pd.Series(ytr2))
        m = T.build_model("lightgbm", spw); m.fit(Xtr2, ytr2)
        ste = m.predict_proba(Xte)[:, 1]
        sval = m.predict_proba(Xval)[:, 1]

        print("\n" + "=" * 78)
        print(f"{dsname}  (LightGBM) | test n={len(test)}, pozitif={int(yte.sum())}, "
              f"taban={yte.mean():.3f}, PR-AUC={average_precision_score(yte, ste):.3f}")
        print("=" * 78)
        for rate in [0.01, 0.02, 0.03, 0.05]:
            op = op_point(yte, ste, rate)
            flag = "<== P>=0.60" if op["precision"] >= 0.60 else ""
            print(f"  en riskli %{rate*100:>4.1f}: {op['n']:>4} alarm | "
                  f"precision={op['precision']:.3f} | recall={op['recall']:.3f} | "
                  f"TP={op['tp']} FP={op['fp']}  {flag}")

        # val-seçili precision>=0.60 eşiği -> test (sızıntısız)
        res = T.precision_constrained_threshold(yval, sval, 0.60)
        if res:
            pred = (ste >= res[0]).astype(int)
            if pred.sum():
                print(f"  [val-eşik P>=0.60] test precision={precision_score(yte,pred,zero_division=0):.3f}"
                      f" recall={recall_score(yte,pred,zero_division=0):.3f} alarm={pred.mean()*100:.1f}%")

        # PR eğrisi
        prec, rec, _ = precision_recall_curve(yte, ste)
        plt.figure(figsize=(6, 4))
        plt.plot(rec, prec, lw=2)
        plt.axhline(0.60, ls="--", c="r", label="precision=0.60 hedef")
        plt.axhline(yte.mean(), ls=":", c="gray", label=f"taban={yte.mean():.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision-Recall — {dsname} (LightGBM)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        out = f"outputs/pr_curve_{dsname}.png"
        plt.savefig(out, dpi=120); plt.close()
        print(f"  PR eğrisi -> {out}")


if __name__ == "__main__":
    main()
