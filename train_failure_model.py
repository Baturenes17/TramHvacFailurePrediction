"""
Tram HVAC Arıza Tahmini — Eğitim & Tahmin Pipeline'ı
====================================================

Amaç: `3_years_data.csv` içindeki `failure_next_30d` etiketini kullanarak bir aracın
önümüzdeki 30 gün içinde HVAC arızası yapıp yapmayacağını tahmin etmek.

Akış:
  1. Veriyi yükle (`;` ayraçlı, `,` ondalık).
  2. Sızıntısız (causal) feature engineering uygula.
  3. Takvim-bazlı 80/20 train/test split (gelecekten geçmişe sızıntı yok).
  4. LightGBM eğit (sınıf dengesizliği için scale_pos_weight; sabit ağaç sayısı,
     --early-stopping ile test üzerinde early stopping opsiyonel).
  5. Test üzerinde ROC-AUC / PR-AUC / classification_report raporla.
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
ALARM_RATE = 0.10                     # Precision-odaklı varsayılan: her dönemde en riskli %10
RANDOM_STATE = 42
TRAIN_FRAC = 0.80                     # Test = kalan %20 (takvim bazlı)
EARLY_STOPPING_ROUNDS = 50

TARGET = f"failure_next_{PREDICTION_HORIZON_DAYS}d"
OTHER_LABEL = "failure_next_7d"       # Sızıntıyı önlemek için özelliklerden çıkarılır
ID_COLS = ["date", "vehicle_id"]
RECALL_TARGETS = [0.90, 0.80, 0.70, 0.60, 0.50]

# Kategorik olarak ele alınacak ham/üretilmiş sütunlar
# NOT: `cevre_temizligi` (temiz/karli/...) ham veride metin kategoriktir; listeye
# dahil değilse get_feature_columns onu sayısal sanır ve SimpleImputer(median)
# 'temiz'i float'a çeviremeyip hata verir.
# `mevsim` (kis/ilkbahar/...) türetilen `season` ile aynı bilgiyi taşıdığı için
# engineer_features içinde düşürülür (tekrarı önlemek için).
CATEGORICAL_FEATURES = [
    "vehicle_type", "weather_type", "season", "cevre_temizligi",
]


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

    # --- Arıza geçmişi (causal) ---
    # `days_since_last_failure` bir arıza olduğunda sıfırlanır; bir önceki güne göre
    # DÜŞTÜYSE o gün bir arıza olmuştur -> failure_event.
    dsf_prev = df.groupby("vehicle_id")["days_since_last_failure"].shift(1)
    df["failure_event"] = (df["days_since_last_failure"] < dsf_prev).astype(int)
    fe = df.groupby("vehicle_id")["failure_event"]
    # Hepsi shift(1): bugünün/geleceğin arızası sayıma GİRMEZ (sızıntı yok).
    df["veh_past_failures"] = fe.transform(lambda s: s.shift(1, fill_value=0).cumsum())
    df["veh_past_obs"] = df.groupby("vehicle_id").cumcount()  # bugüne kadarki gözlem günü
    df["veh_past_failure_rate"] = (
        df["veh_past_failures"] / df["veh_past_obs"].replace(0, np.nan)
    ).fillna(0.0)
    df["veh_failures_last_90d"] = fe.transform(
        lambda s: s.shift(1, fill_value=0).rolling(90, min_periods=1).sum()
    )

    # --- Birikimli hava stresi (tarih bazlı; temp/weather tüm filoda aynı) ---
    # Tek tarih serisi üzerinden hesaplayıp geri birleştir. Pencereler aynı-gün
    # dahil (geçmiş+bugünün hava DURUMU gözlemlenmiştir -> sızıntı yok).
    daily_w = (
        df.drop_duplicates("date").set_index("date").sort_index()["temp"]
    )
    roll = pd.DataFrame(index=daily_w.index)
    roll["temp_7d_mean"] = daily_w.rolling(7, min_periods=1).mean()
    roll["temp_7d_max"] = daily_w.rolling(7, min_periods=1).max()
    roll["temp_30d_max"] = daily_w.rolling(30, min_periods=1).max()
    roll["hot_days_7d"] = (daily_w >= 30).rolling(7, min_periods=1).sum()
    roll["hot_days_14d"] = (daily_w >= 30).rolling(14, min_periods=1).sum()
    roll["cold_days_7d"] = (daily_w <= 0).rolling(7, min_periods=1).sum()
    df = df.merge(roll, left_on="date", right_index=True, how="left")

    # --- Hava tipi (WMO kodu) ayrıştırması: ordinal yerine anlamlı bayraklar ---
    wt = df["weather_type"]
    df["is_precip"] = (wt >= 51).astype(int)   # 51+ : çiseleme/yağmur/kar
    df["is_rain"] = wt.between(61, 69).astype(int)
    df["is_snow"] = wt.between(71, 79).astype(int)

    # --- Bakım gecikmesi / kullanım yoğunluğu etkileşimleri (aynı-gün bilinir) ---
    df["km_per_day_since_maint"] = df["km_since_last_maintenance"] / (
        df["days_since_last_maintenance"] + eps
    )
    df["age_x_km_since_maint"] = df["vehicle_age"] * df["km_since_last_maintenance"]
    df["age_x_km30"] = df["vehicle_age"] * df["km_last_30d"]
    df["km_today_vs_7davg"] = df["km_today"] / (df["km_last_7d"] / 7 + eps)

    # --- Arıza geçmişi dönüşümleri (days_since_last_failure aynı-gün gözlemli) ---
    df["log_days_since_failure"] = np.log1p(df["days_since_last_failure"])
    df["recent_failure_30d"] = (df["days_since_last_failure"] <= 30).astype(int)
    df["recent_failure_90d"] = (df["days_since_last_failure"] <= 90).astype(int)

    # Yardımcı/sürüklenen kolonları düşür:
    #  - failure_event: bugünü gösterir -> sızıntı, özellik OLAMAZ.
    #  - veh_past_obs & veh_past_failures: zamanla monoton büyür (time-index proxy);
    #    zaman-bazlı split'te dağılım kayması yaratıp genellemeyi bozar. Sadece
    #    durağan veh_past_failure_rate ve veh_failures_last_90d özellik olarak kalır.
    df = df.drop(columns=["failure_event", "veh_past_obs", "veh_past_failures"])

    # `mevsim` ham sütunu, yukarıda aydan türetilen `season` ile aynı bilgiyi taşır;
    # tekrarı önlemek için düşürülür (varsa).
    df = df.drop(columns=["mevsim"], errors="ignore")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Modelin kullanacağı özellik sütunları: kimlik, hedef ve sızıntı kolonları hariç."""
    exclude = set(ID_COLS) | {TARGET, OTHER_LABEL}
    return [c for c in df.columns if c not in exclude]


# --------------------------------------------------------------------------- #
# 4. Takvim bazlı bölme
# --------------------------------------------------------------------------- #
def time_based_split(df: pd.DataFrame, test_end: pd.Timestamp | None = None):
    """Tarih aralığına göre 80/20 train/test (takvim bazlı). Kronolojik kesim — sızıntı yok.

    test_end verilirse, test setinde bu tarihten (dahil) SONRAKİ satırlar atılır.
    Veri sonundaki günlerde failure_next_30d için tam 30 günlük ileri pencere
    bulunmadığından (etiket eksik/güvenilmez), kuyruk böyle kırpılabilir.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True)
    dmin, dmax = df_sorted["date"].min(), df_sorted["date"].max()
    span = dmax - dmin
    train_end = dmin + span * TRAIN_FRAC
    train = df_sorted[df_sorted["date"] <= train_end]
    test = df_sorted[df_sorted["date"] > train_end]
    if test_end is not None:
        test = test[test["date"] <= test_end]
    return train, test


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


MODEL_CHOICES = ["lightgbm", "logreg", "xgboost", "catboost", "ensemble"]
# TreeExplainer (SHAP) ve early stopping yalnızca bu ağaç tabanlı modeller için anlamlı.
TREE_MODELS = {"lightgbm", "xgboost", "catboost"}

# Kısıtlı/regularize LightGBM parametreleri (--regularized). Varsayılan ağaçlar train'i
# ezberliyordu (train precision ~0.86, test ~0.36). Bu sığ/cezalı ağaçlar ezberi kırar;
# düşük alarm oranlarında test precision'ı belirgin yükseltir (örn. %1 alarmda ~0.61).
REGULARIZED_LGBM_PARAMS = dict(
    n_estimators=120,
    learning_rate=0.02,
    num_leaves=8,
    max_depth=3,
    min_child_samples=200,
    reg_lambda=5.0,
    subsample=0.7,
    colsample_bytree=0.7,
)


def _build_lightgbm(scale_pos_weight: float, params: dict | None = None):
    base = dict(
        n_estimators=200,
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


def _build_xgboost(scale_pos_weight: float, params: dict | None = None):
    from xgboost import XGBClassifier

    base = dict(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if params:
        base.update(params)
    return XGBClassifier(**base)


def _build_catboost(scale_pos_weight: float, params: dict | None = None):
    from catboost import CatBoostClassifier

    base = dict(
        iterations=300,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        scale_pos_weight=scale_pos_weight,
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,  # catboost_info/ klasörü oluşturma
    )
    if params:
        base.update(params)
    return CatBoostClassifier(**base)


def build_model(model_type: str, scale_pos_weight: float, params: dict | None = None):
    """Seçilen modeli kur.

    - lightgbm: gradient-boosted ağaçlar (scale_pos_weight ile dengelenir).
    - logreg:   ölçeklenmiş Lojistik Regresyon (class_weight='balanced').
      Benchmark'ta bu zayıf-lineer sinyalde ağaç modellerini geçti; bu yüzden
      precision'ı bir tık yükseltmek için seçenek olarak sunulur.
    - xgboost / catboost: alternatif gradient-boosting kütüphaneleri; aynı
      scale_pos_weight dengelemesiyle. SHAP ve early stopping desteklenir.
    - ensemble: lightgbm + xgboost + catboost olasılıklarının soft-voting
      ortalaması (her biri scale_pos_weight ile dengeli). Tek modellerin
      gürültüsünü ortalayarak daha kararlı skor amaçlar.

    `params` yalnızca tekil ağaç modellerine uygulanır (ensemble alt modelleri
    varsayılan parametrelerle kurulur).
    """
    if model_type == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    if model_type == "lightgbm":
        return _build_lightgbm(scale_pos_weight, params)
    if model_type == "xgboost":
        return _build_xgboost(scale_pos_weight, params)
    if model_type == "catboost":
        return _build_catboost(scale_pos_weight, params)
    if model_type == "ensemble":
        from sklearn.ensemble import VotingClassifier

        return VotingClassifier(
            estimators=[
                ("lgbm", _build_lightgbm(scale_pos_weight)),
                ("xgb", _build_xgboost(scale_pos_weight)),
                ("cat", _build_catboost(scale_pos_weight)),
            ],
            voting="soft",
        )

    raise ValueError(f"Bilinmeyen model_type: {model_type}")


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


def precision_constrained_threshold(y_true, scores, target_precision: float):
    """precision >= target altında recall'ı maksimize eden (en düşük uygun) eşik.
    Dönüş: (eşik, precision, recall) veya hedef hiç tutmazsa None."""
    ths = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 400)))
    for t in ths:  # artan eşik -> ilk hedefi tutan en yüksek recall'ı verir
        pred = (scores >= t).astype(int)
        if pred.sum() == 0:
            continue
        p = precision_score(y_true, pred, zero_division=0)
        if p >= target_precision:
            r = recall_score(y_true, pred, zero_division=0)
            return (float(t), p, r)
    return None


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
def tune_optuna(df_train, feature_cols, n_trials):
    import optuna
    from sklearn.model_selection import TimeSeriesSplit

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df_sorted = df_train.sort_values("date").reset_index(drop=True)
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
            model = build_model("lightgbm", spw, params)
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
    parser.add_argument("--model", choices=MODEL_CHOICES, default="lightgbm",
                        help="Sınıflandırıcı: 'lightgbm' (varsayılan), 'logreg' "
                             "(ölçeklenmiş Lojistik Regresyon — bu veride biraz daha "
                             "yüksek precision), 'xgboost', 'catboost' veya 'ensemble' "
                             "(lightgbm+xgboost+catboost soft-voting)")
    parser.add_argument("--regularized", action="store_true",
                        help="Kısıtlı/regularize LightGBM parametrelerini kullan "
                             "(sığ ağaçlar; overfit'i azaltır, düşük --alarm-rate ile "
                             "test precision'ı yükseltir). Yalnız lightgbm.")
    parser.add_argument("--tune", action="store_true", help="Optuna hiperparametre araması (yalnız lightgbm)")
    parser.add_argument("--trials", type=int, default=40, help="Optuna deneme sayısı")
    parser.add_argument("--shap", action="store_true", help="SHAP özellik önemi üret")
    parser.add_argument("--alarm-rate", type=float, default=ALARM_RATE,
                        help="Alarm-oranı eşiği için pozitif yüzdesi (0-1)")
    parser.add_argument("--threshold-mode", choices=["alarm", "fixed", "precision"],
                        default="alarm", help="Eşik seçim modu")
    parser.add_argument("--target-precision", type=float, default=0.50,
                        help="'precision' modunda hedeflenen minimum precision")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Test üzerinde early stopping kullan "
                             "(varsayılan: kapalı — sabit ağaç sayısı; "
                             "test'e baktığı için önerilmez)")
    parser.add_argument("--smote", action="store_true",
                        help="Yalnız TRAIN setine SMOTENC uygula: arıza sınıfını "
                             "(failure_next_30d=1) çoğalt. test'e dokunulmaz.")
    parser.add_argument("--smote-ratio", default="auto",
                        help="SMOTE sampling_strategy: 'auto' = 1:1 tam denge "
                             "(varsayılan). Daha ılımlı için ondalık ver: ör. 0.5 "
                             "(1 arıza : 2 normal).")
    parser.add_argument("--predict", default=None,
                        help="Eğitim sonrası skorlanacak ek CSV yolu")
    parser.add_argument("--test-end", default=None,
                        help="Test setinde bu tarihten (dahil) sonrasını dışla. "
                             "Ör: 2025-12-01 (etiketi eksik kuyruk günlerini at).")
    parser.add_argument("--test-neg-keep", type=float, default=1.0,
                        help="Test negatiflerinin (failure=0) tutulacak oranı [0-1]. "
                             "Precision deneyi: <1 verilince negatifler rastgele "
                             "altörneklenir (train'e DOKUNULMAZ).")
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
    test_end = pd.to_datetime(args.test_end) if args.test_end else None
    train, test = time_based_split(df, test_end=test_end)
    if test_end is not None:
        print(f"[test-end] Test {test_end.date()} sonrası dışlandı.")

    # 4b) (Opsiyonel) Precision deneyi: test negatiflerini altörnekle.
    #     Pozitifler (failure=1) korunur; negatiflerin yalnız bir kısmı tutulur.
    #     Bu, test sınıf dağılımını yapay olarak dengeler -> daha az FP -> precision
    #     yükselir. Train'e dokunulmaz; gerçek operasyon precision'ı değil, sınıf
    #     dengesi precision'ı nasıl etkiliyor onu görmek için bir what-if'tir.
    if args.test_neg_keep < 1.0:
        pos = test[test[TARGET] == 1]
        neg = test[test[TARGET] == 0]
        neg_kept = neg.sample(frac=args.test_neg_keep, random_state=RANDOM_STATE)
        n_neg_before = len(neg)
        test = pd.concat([pos, neg_kept]).sort_values("date").reset_index(drop=True)
        print(f"[test-neg-keep={args.test_neg_keep:.2f}] test negatif: "
              f"{n_neg_before} -> {len(neg_kept)} | pozitif: {len(pos)} (sabit) | "
              f"yeni 1-oranı: {len(pos)/len(test)*100:.1f}%")

    print(f"Train size: {len(train)}  dates: {_fmt_range(train)}")
    print(f"Test  size: {len(test)}  dates: {_fmt_range(test)}")

    print(f"Model: {args.model}")

    # (Opsiyonel) Optuna — yalnız LightGBM
    best_params = None
    if args.tune:
        if args.model != "lightgbm":
            print(f"[Uyarı] --tune yalnız lightgbm için geçerli; "
                  f"'{args.model}' modelinde atlanıyor.")
        else:
            best_params = tune_optuna(train, feature_cols, args.trials)

    # (Opsiyonel) Kısıtlı/regularize LightGBM parametreleri (--tune verilmemişse).
    if args.regularized:
        if args.model != "lightgbm":
            print(f"[Uyarı] --regularized yalnız lightgbm için geçerli; "
                  f"'{args.model}' modelinde atlanıyor.")
        elif best_params is not None:
            print("[Uyarı] --tune ile --regularized birlikte verildi; "
                  "Optuna parametreleri kullanılıyor.")
        else:
            best_params = REGULARIZED_LGBM_PARAMS
            print(f"[regularized] Kısıtlı LightGBM parametreleri: {best_params}")

    # 5) Eğitim — preprocessor + LightGBM (early stopping test üzerinde)
    prep = build_preprocessor(feature_cols)
    Xtr = prep.fit_transform(train[feature_cols])
    Xte = prep.transform(test[feature_cols])
    ytr = train[TARGET].values
    yte = test[TARGET].values

    # 5b) (Opsiyonel) SMOTE — YALNIZ train'e. Arıza sınıfını (failure_next_30d=1)
    #     sentetik örneklerle çoğaltır. test gerçek dağılımıyla kalır (sızıntı yok).
    #     ÖNEMLİ: preprocessor'dan SONRA uygulanır. Çünkü SMOTENC NaN kabul etmez;
    #     ham feature'larda eksikler var ve bunlar SimpleImputer ile burada dolduruldu.
    #     ColumnTransformer çıktısında kolon sırası [sayısal..., kategorik...] olduğu
    #     için kategorik (ordinal-encoded) kolonlar en sonda yer alır.
    if args.smote:
        from imblearn.over_sampling import SMOTENC

        n_num = len([c for c in feature_cols if c not in CATEGORICAL_FEATURES])
        n_cat = len([c for c in feature_cols if c in CATEGORICAL_FEATURES])
        cat_idx = list(range(n_num, n_num + n_cat))  # encode'lu kategorikler en sonda
        try:
            strat = float(args.smote_ratio)
        except ValueError:
            strat = args.smote_ratio  # "auto" = 1:1
        sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=strat,
                     random_state=RANDOM_STATE)
        n_before, pos_before = len(Xtr), int((ytr == 1).sum())
        Xtr, ytr = sm.fit_resample(Xtr, ytr)
        pos_after = int((ytr == 1).sum())
        print(f"[SMOTE] train {n_before} -> {len(Xtr)} satır | "
              f"arıza(1): {pos_before} -> {pos_after} "
              f"(+{pos_after - pos_before} sentetik) | "
              f"1-oranı: {pos_before/n_before*100:.1f}% -> "
              f"{pos_after/len(Xtr)*100:.1f}%")

    # NOT: SMOTE dengeyi düzelttiğinde scale_pos_weight'i SMOTE SONRASI y'den
    # hesaplıyoruz (yoksa hem oversampling hem ağırlık -> çifte düzeltme).
    spw = compute_scale_pos_weight(pd.Series(ytr))

    model = build_model(args.model, spw, best_params)
    best_iter = None
    if args.early_stopping and args.model != "lightgbm":
        print(f"[Uyarı] --early-stopping yalnız lightgbm için uygulanır; "
              f"'{args.model}' modelinde atlanıyor.")
    if args.model == "lightgbm" and args.early_stopping:
        # NOT: Ayrı validation seti kaldırıldı; early stopping istenirse eval seti
        # olarak test kullanılır. Bu, test metriklerini hafifçe iyimser yapar —
        # bu yüzden early stopping varsayılan KAPALIDIR ve önerilmez (alarm-oranı
        # eşiği + sabit ağaç sayısı tercih edilir).
        model.fit(
            Xtr, ytr,
            eval_set=[(Xte, yte)],
            eval_metric="auc",
            callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(0)],
        )
        best_iter = model.best_iteration_ or model.n_estimators
        print(f"[LightGBM] best iteration (early stopping, eval=test): {best_iter}")
    else:
        model.fit(Xtr, ytr)
        if args.model == "lightgbm":
            best_iter = model.n_estimators
            print(f"[LightGBM] sabit ağaç sayısı: {best_iter} (early stopping kapalı)")
        elif args.model == "logreg":
            print("[logreg] eğitildi (sınıf dengesizliği class_weight='balanced' ile)")
        else:
            print(f"[{args.model}] eğitildi (sınıf dengesizliği scale_pos_weight ile)")

    # 6) Skorlar + eşikler
    #    Validation seti kaldırıldı; sabit-eşik modları (fixed/precision) için
    #    referans olarak TRAIN skorları kullanılır (test'e bakarak eşik seçmek
    #    sızıntı olurdu). Train üzerinde seçilen eşikler iyimser olabilir; bu
    #    yüzden alarm-oranı modu (her split kendi quantile'ı) varsayılan ve önerilendir.
    train_scores = model.predict_proba(Xtr)[:, 1]
    test_scores = model.predict_proba(Xte)[:, 1]

    op = recall_constrained_threshold(ytr, train_scores, 0.80)
    op_t = op[0] if op else float("nan")
    f1opt = f1_optimal_threshold(ytr, train_scores)

    if args.threshold_mode == "alarm":
        # Her split kendi quantile'ını kullanır (train ve test için ayrı eşik).
        train_thr = alarm_rate_threshold(train_scores, args.alarm_rate)
        test_thr = alarm_rate_threshold(test_scores, args.alarm_rate)
        print(f"Eşik modu — alarm-oranı (en riskli %{args.alarm_rate*100:.0f}) "
              f"| referans (train): op(recall>=0.80)={op_t:.3f}, F1-opt={f1opt:.3f}")
    elif args.threshold_mode == "precision":
        res = precision_constrained_threshold(ytr, train_scores, args.target_precision)
        if res:
            train_thr = test_thr = res[0]
            print(f"Eşik modu — precision-hedef (train precision>={args.target_precision:.2f}) "
                  f"eşik={res[0]:.3f} | train'de precision={res[1]:.3f}, recall={res[2]:.3f}")
        else:
            # Hedefe ulaşılamadı: en yüksek skorlu %10'u işaretle (en konservatif).
            train_thr = test_thr = alarm_rate_threshold(train_scores, 0.10)
            print(f"Eşik modu — precision-hedef ({args.target_precision:.2f}) train'de tutmadı; "
                  f"geri dönüş: en riskli %10, eşik={test_thr:.3f}")
    else:
        train_thr = test_thr = op_t
        print(f"Eşik modu — sabit eşik (train recall>=0.80)={op_t:.3f} "
              f"| referans: F1-opt={f1opt:.3f}")

    # 7) Per-split raporlar (train + test)
    print("\n--- TRAIN ---")
    train_roc = report_split("Train", ytr, train_scores, train_thr)
    precision_recall_tradeoff(ytr, train_scores)

    print("\n--- TEST ---")
    test_roc = report_split("Test", yte, test_scores, test_thr)
    precision_recall_tradeoff(yte, test_scores)

    # 9) SHAP (TreeExplainer — yalnız ağaç tabanlı modeller)
    if args.shap:
        if args.model not in TREE_MODELS:
            print(f"[Uyarı] --shap (TreeExplainer) yalnız ağaç modelleri için "
                  f"({', '.join(sorted(TREE_MODELS))}); '{args.model}' atlanıyor.")
        else:
            sample = test[feature_cols].sample(min(2000, len(test)), random_state=RANDOM_STATE)
            run_shap(prep, model, sample, args.out)

    print("\nModel training completed.")
    print("Train ROC-AUC:", train_roc)
    print("Test ROC-AUC:", test_roc)

    # 10) Final model: TÜM veriyle yeniden fit -> güncel skorlama
    spw_full = compute_scale_pos_weight(df[TARGET])
    final_params = dict(best_params or {})
    if args.model == "lightgbm" and best_iter is not None:
        final_params["n_estimators"] = int(best_iter)
    final_prep = build_preprocessor(feature_cols)
    Xall = final_prep.fit_transform(df[feature_cols])
    final_model = build_model(args.model, spw_full, final_params)
    final_model.fit(Xall, df[TARGET].values)

    score_latest(df, final_prep, final_model, feature_cols, args.alarm_rate, args.out)

    if args.predict:
        predict_file(args.predict, final_prep, final_model, feature_cols,
                     args.alarm_rate, args.out)


if __name__ == "__main__":
    main()