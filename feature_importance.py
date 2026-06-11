"""LightGBM gain-bazlı feature importance — train_failure_model fonksiyonlarını
yeniden kullanır. Kullanım:
    python feature_importance.py --data 3_years_data_clean.csv
"""
import argparse

import numpy as np
import pandas as pd

from train_failure_model import (
    DEFAULT_DATA, TARGET,
    load_data, engineer_features, get_feature_columns, time_based_split,
    build_preprocessor, build_model, compute_scale_pos_weight,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    df = engineer_features(load_data(args.data))
    feature_cols = get_feature_columns(df)
    train, _ = time_based_split(df)

    prep = build_preprocessor(feature_cols)
    Xtr = prep.fit_transform(train[feature_cols])
    ytr = train[TARGET].values

    spw = compute_scale_pos_weight(pd.Series(ytr))
    model = build_model("lightgbm", spw, None)
    model.fit(Xtr, ytr)

    names = prep.get_feature_names_out()
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")

    imp = (
        pd.DataFrame({"feature": names, "gain": gain, "split": split})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp["gain_pct"] = imp["gain"] / imp["gain"].sum() * 100

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(f"\nToplam özellik: {len(imp)} | Top {args.top} (gain'e göre):\n")
    print(imp.head(args.top).to_string(index=False,
          formatters={"gain": "{:.0f}".format, "gain_pct": "{:.1f}%".format}))

    out = "outputs/feature_importance.csv"
    imp.to_csv(out, index=False)
    print(f"\nTam liste -> {out}")


if __name__ == "__main__":
    main()
