"""
HEDEF: TEST precision >= 0.60 ulaşılabilir mi? Hangi recall / alarm oranıyla?
============================================================================
İki şey ölçülür (her veri seti × model):

  1) ORACLE TAVAN (test PR eğrisi): test'in kendisinde precision>=0.60'ı tutan
     EN YÜKSEK recall. Bu, "mümkün mü?" üst sınırıdır (iyimser, eşik test'e bakılarak).

  2) DÜRÜST (sızıntısız): eşik VALIDATION üzerinde seçilir (train'in son %25'i,
     kronolojik), precision>=hedef olacak şekilde; sonra TEST'te precision/recall/
     alarm-oranı raporlanır. Sunumda savunulacak olan budur.

Ayrıca test'te top-%1/%2/%5 alarm oranlarında precision tablosu.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import train_failure_model as T

warnings.filterwarnings("ignore")

DATASETS = {
    "full": "3_years_data.csv",
    "no2025": "3_years_data_no2025.csv",
    "until_3871": "3_years_data_until_3871_clean.csv",
    "no2025_3871": "3_years_data_no2025_3871_clean.csv",
}
MODELS = ["lightgbm", "lightgbm-reg", "logreg", "catboost", "ensemble"]
TARGET_P = 0.60


def build(model_type, spw):
    if model_type == "lightgbm-reg":
        return T.build_model("lightgbm", spw, T.REGULARIZED_LGBM_PARAMS)
    return T.build_model(model_type, spw)


def max_recall_at_precision(y, s, target):
    """test PR eğrisinde precision>=target'ı tutan en yüksek recall + o eşik."""
    prec, rec, thr = precision_recall_curve(y, s)
    ok = prec[:-1] >= target  # son nokta eşiksiz
    if not ok.any():
        return 0.0, None, float(prec.max())
    i = np.argmax(rec[:-1] * ok)  # ok olanlar içinde en yüksek recall
    return float(rec[i]), float(thr[i]), float(prec.max())


def prec_at_alarm(y, s, rate):
    thr = np.quantile(s, 1 - rate)
    pred = (s >= thr).astype(int)
    if pred.sum() == 0:
        return float("nan"), float("nan")
    return precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0)


def main():
    for dsname, path in DATASETS.items():
        df = T.engineer_features(T.load_data(path))
        feat = T.get_feature_columns(df)
        train, test = T.time_based_split(df)
        # train'i kronolojik train2 / val (son %25) olarak böl
        train = train.sort_values("date").reset_index(drop=True)
        cut = int(len(train) * 0.75)
        tr2, val = train.iloc[:cut], train.iloc[cut:]

        prep = T.build_preprocessor(feat)
        Xtr2 = prep.fit_transform(tr2[feat]); ytr2 = tr2[T.TARGET].values
        Xval = prep.transform(val[feat]);     yval = val[T.TARGET].values
        Xte = prep.transform(test[feat]);     yte = test[T.TARGET].values
        base = yte.mean()

        print("\n" + "=" * 88)
        print(f"VERİ SETİ: {dsname}  (test taban arıza oranı={base:.3f}, "
              f"test n={len(test)}, pozitif={int(yte.sum())})")
        print("=" * 88)
        print(f"{'Model':<13}{'ORACLE tavan':<22}{'DÜRÜST (val-eşik)':<34}{'top%1 / %2 / %5 prec':<22}")
        print(f"{'':<13}{'(maxRec @P>=.60)':<22}{'testP / testR / alarm%':<34}")
        print("-" * 88)

        for mt in MODELS:
            spw = T.compute_scale_pos_weight(pd.Series(ytr2))
            m = build(mt, spw); m.fit(Xtr2, ytr2)
            sval = m.predict_proba(Xval)[:, 1]
            ste = m.predict_proba(Xte)[:, 1]

            # 1) ORACLE tavan (test'e bakarak)
            orec, _, pmax = max_recall_at_precision(yte, ste, TARGET_P)

            # 2) DÜRÜST: val'da precision>=hedef en düşük eşik (en yüksek recall)
            res = T.precision_constrained_threshold(yval, sval, TARGET_P)
            if res:
                thr = res[0]
                pred = (ste >= thr).astype(int)
                if pred.sum() > 0:
                    tp = precision_score(yte, pred, zero_division=0)
                    tr = recall_score(yte, pred, zero_division=0)
                    alarm = pred.mean()
                    honest = f"{tp:.3f} / {tr:.3f} / {alarm*100:.1f}%"
                else:
                    honest = "val-eşik test'te 0 alarm"
            else:
                honest = f"val'da P>=.60 tutmadı (max {sval.max():.2f})"

            # 3) test top-%1/%2/%5 precision
            p1 = prec_at_alarm(yte, ste, 0.01)[0]
            p2 = prec_at_alarm(yte, ste, 0.02)[0]
            p5 = prec_at_alarm(yte, ste, 0.05)[0]

            oracle = f"R={orec:.3f} (Pmax={pmax:.2f})"
            print(f"{mt:<13}{oracle:<22}{honest:<34}"
                  f"{p1:.2f} / {p2:.2f} / {p5:.2f}")


if __name__ == "__main__":
    main()
