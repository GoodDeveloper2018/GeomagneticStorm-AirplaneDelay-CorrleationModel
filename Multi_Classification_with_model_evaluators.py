# ============================================
# Geomagnetic Storms → Aviation Delay Modeling
# Physics-informed features + CV + tuning
# ============================================

import pandas as pd
import numpy as np
import re

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    GroupKFold,
    GridSearchCV
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

# -------------------------------------------------
# 0) Paths (edit these to match your local machine)
# -------------------------------------------------
flight_data_path = r"C:\Users\arshp\OneDrive\Desktop\Coding Work\Terra Research\JFK-LAX flights 2023 Model.csv"
geomagnetic_storm_duration_data_path = r"C:\Users\arshp\OneDrive\Desktop\Coding Work\Terra Research\Geomagnetic Storms - duration.xlsx"
geomagnetic_storm_data_points_path = r"C:\Users\arshp\OneDrive\Desktop\Coding Work\Terra Research\Geomagnetic-Storm_Data.csv"


# -------------------------------------------------
# 1) Helper functions
# -------------------------------------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[()]", "", regex=True)
        .str.replace("__+", "_", regex=True)
    )
    return df


def first_number(s):
    """Extract first numeric token from a string (for Kp, Speed…)."""
    if pd.isna(s):
        return np.nan
    s = str(s).replace("\u2212", "-")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def last_number(s):
    """Extract last numeric token (for Dst, Bt, Bz…)."""
    if pd.isna(s):
        return np.nan
    s = str(s).replace("\u2212", "-")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else np.nan


def parse_time_series(s: pd.Series) -> pd.Series:
    """Robust parsing of time strings into Timestamps."""
    s = s.astype(str).str.strip()
    t = pd.to_datetime(s, format='%H:%M', errors='coerce')
    m = t.isna()
    t.loc[m] = pd.to_datetime(s[m], format='%I:%M %p', errors='coerce')
    m = t.isna()
    t.loc[m] = pd.to_datetime(s[m], format='%H:%M:%S', errors='coerce')
    return t


def evaluate_model(name, model, X_test, y_test, prob_method="predict_proba"):
    """Standard evaluation + plots."""
    if prob_method == "predict_proba":
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # For models with decision_function only
        y_score = model.decision_function(X_test)
        # normalize to [0,1] for ROC/PR
        y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3))
    roc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC: {roc:.3f} | PR-AUC: {ap:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix — {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f'ROC Curve — {name}')
    plt.tight_layout()
    plt.show()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title(f'Precision–Recall — {name}')
    plt.tight_layout()
    plt.show()

    return roc, ap


# -------------------------------------------------
# 2) Load / clean / merge datasets
# -------------------------------------------------
flight = pd.read_csv(flight_data_path)
storm_monthly = pd.read_excel(geomagnetic_storm_duration_data_path)
storm_daily = pd.read_csv(geomagnetic_storm_data_points_path)

flight = clean_cols(flight)
storm_monthly = clean_cols(storm_monthly)
storm_daily = clean_cols(storm_daily)

# --- Flight data ---
# Normalize key columns
if {'Date', 'Time', 'Total_Delay'}.issubset(flight.columns):
    flight = flight.rename(columns={'Date': 'date', 'Time': 'time', 'Total_Delay': 'delay_minutes'})
elif {'date', 'time', 'total_delay'}.issubset(flight.columns):
    flight = flight.rename(columns={'total_delay': 'delay_minutes'})
else:
    raise ValueError("Flight CSV missing expected (Date/Time/Total_Delay) or (date/time/total_delay).")

flight['date'] = pd.to_datetime(flight['date'], errors='coerce').dt.date
flight['delay_minutes'] = pd.to_numeric(flight['delay_minutes'], errors='coerce')

# Drop irrelevant columns if present
drop_cols = [
    'Terminal', 'Call_Sign', 'Marketing_Airline', 'General_Aircraft_Desc',
    'Max_Takeoff_Wt_Lbs', 'Max_Landing_Wt_Lbs', 'Intl/_Dom', 'Total_Seats',
    'Total_Taxi_Time', 'Direction', 'PA_Airport', 'Non-PA_Airport'
]
flight = flight.drop(columns=[c for c in drop_cols if c in flight.columns], errors='ignore')

flight = flight.dropna(subset=['date', 'delay_minutes'])
print("Flight rows (valid date & delay):", len(flight))

# --- Monthly storm duration → month key and aggregates ---
if {'YYYY', 'MM'}.issubset(storm_monthly.columns):
    storm_monthly['month'] = pd.to_datetime(
        storm_monthly['YYYY'].astype(int).astype(str) + '-' +
        storm_monthly['MM'].astype(int).astype(str).str.zfill(2) + '-01'
    ).dt.to_period('M')
elif 'date' in storm_monthly.columns:
    storm_monthly['month'] = pd.to_datetime(storm_monthly['date'], errors='coerce').dt.to_period('M')
else:
    raise ValueError("Monthly table needs YYYY/MM or date.")

disturbed_cols = [c for c in storm_monthly.columns if c.lower().startswith('d')]
quiet_cols = [c for c in storm_monthly.columns if c.lower().startswith('q')]

if disturbed_cols:
    vals_d = storm_monthly[disturbed_cols].apply(pd.to_numeric, errors='coerce')
    has_any_d = vals_d.notna().sum(axis=1) > 0
    storm_monthly['average_disturbed'] = np.where(has_any_d, vals_d.mean(axis=1), np.nan)

if quiet_cols:
    vals_q = storm_monthly[quiet_cols].apply(pd.to_numeric, errors='coerce')
    has_any_q = vals_q.notna().sum(axis=1) > 0
    storm_monthly['average_quiet'] = np.where(has_any_q, vals_q.mean(axis=1), np.nan)

storm_monthly = storm_monthly[['month', 'average_disturbed', 'average_quiet']].drop_duplicates()
print("Monthly storm months:", storm_monthly['month'].nunique())

# --- Daily storm metrics ---
name_map = {}
for col in list(storm_daily.columns):
    c = col.lower().replace(" ", "_").replace("-", "_")
    if "kp" in c and "max" in c:
        name_map[col] = "Kp_max"
    elif c in ("kp", "k_p"):
        name_map[col] = "Kp_max"
    elif c.startswith("ap"):
        name_map[col] = "Ap"
    elif c.startswith("dst"):
        name_map[col] = "Dst"
    elif "imf" in c and "bt" in c:
        name_map[col] = "IMF_Bt"
    elif "imf" in c and "bz" in c:
        name_map[col] = "IMF_Bz"
    elif "speed" in c or "km_s" in c:
        name_map[col] = "Speed"

storm_daily = storm_daily.rename(columns=name_map)

if 'date' in storm_daily.columns:
    storm_daily['date'] = pd.to_datetime(storm_daily['date'], errors='coerce').dt.date
elif 'Date' in storm_daily.columns:
    storm_daily['date'] = pd.to_datetime(storm_daily['Date'], errors='coerce').dt.date
else:
    raise ValueError("Daily table missing date/Date.")

for col in ['Dst', 'Ap', 'Kp_max', 'Speed', 'IMF_Bt', 'IMF_Bz']:
    if col in storm_daily.columns:
        if col in ['Kp_max', 'Speed']:
            storm_daily[col] = storm_daily[col].map(first_number)
        else:
            storm_daily[col] = storm_daily[col].map(last_number)

storm_daily = storm_daily[['date', 'Dst', 'Ap', 'Kp_max', 'IMF_Bt', 'IMF_Bz', 'Speed']].dropna(subset=['date'])
print("Daily storm rows:", len(storm_daily))

# --- Merge flight & storms ---
df = pd.merge(flight, storm_daily, on='date', how='inner')
print("After flight×daily merge:", df.shape)

df['month'] = pd.to_datetime(df['date'], errors='coerce').dt.to_period('M')
df = pd.merge(df, storm_monthly, on='month', how='left')
print("After +monthly merge:", df.shape)

# -------------------------------------------------
# 3) Time-of-day encoding + physics-informed flags
# -------------------------------------------------
# Time-of-day features
t_obj = parse_time_series(df['time'])
df['hour'] = t_obj.dt.hour.fillna(0)
df['minute'] = t_obj.dt.minute.fillna(0)
df['time_minutes'] = df['hour'] * 60 + df['minute']
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

# Physics-informed storm flags
if 'Kp_max' in df.columns:
    df['high_kp'] = (df['Kp_max'] >= 6).astype(int)      # storm threshold
else:
    df['high_kp'] = 0

if 'Dst' in df.columns:
    df['strong_storm'] = (df['Dst'] <= -70).astype(int)  # stronger ring current
else:
    df['strong_storm'] = 0

if 'IMF_Bz' in df.columns:
    df['bz_neg'] = (df['IMF_Bz'] <= -10).astype(int)     # strong southward Bz
else:
    df['bz_neg'] = 0

df['night_high_kp'] = df['night'] * df['high_kp']
df['night_bz_neg'] = df['night'] * df['bz_neg']

# Simple electric field / coupling proxies
if {'Speed', 'IMF_Bz'}.issubset(df.columns):
    df['Ey'] = -(df['Speed'] * df['IMF_Bz'])  # convective electric field proxy
else:
    df['Ey'] = 0.0

if {'IMF_Bt', 'Speed'}.issubset(df.columns):
    Bt = df['IMF_Bt'].clip(lower=0)
    Vz = df['Speed'].clip(lower=0)
    df['storm_coupling'] = (Vz ** (4/3)) * (Bt ** (2/3))
else:
    df['storm_coupling'] = 0.0

# Fill missing numeric storm features with medians
for c in ['Dst', 'Ap', 'Kp_max', 'IMF_Bt', 'IMF_Bz', 'Speed',
          'average_disturbed', 'average_quiet', 'Ey', 'storm_coupling']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

# -------------------------------------------------
# 4) Label engineering: richer delay targets
# -------------------------------------------------
df['delay_minutes'] = pd.to_numeric(df['delay_minutes'], errors='coerce')
df = df.dropna(subset=['delay_minutes'])

# Binary: any delay vs no delay
df['delay_binary'] = (df['delay_minutes'] > 0).astype(int)

# More severe delays (e.g. ≥30 minutes)
df['big_delay_30'] = (df['delay_minutes'] >= 30).astype(int)

# Multi-class severity (you can use later if you want)
bins = [-1, 0, 14, 59, 10_000]
labels = ['on_time', 'minor', 'moderate', 'major']
df['delay_severity'] = pd.cut(df['delay_minutes'], bins=bins, labels=labels)

print("Delay_binary distribution:\n", df['delay_binary'].value_counts())
print("Big_delay_30 distribution:\n", df['big_delay_30'].value_counts())
print("Delay_severity distribution:\n", df['delay_severity'].value_counts())

# Choose which target to model:
LABEL_COL = "delay_binary"   # change to 'big_delay_30' if you want to focus on major delays

# -------------------------------------------------
# 5) Build feature matrix X & labels y
# -------------------------------------------------
storm_features = [
    'Dst', 'Ap', 'Kp_max', 'IMF_Bt', 'IMF_Bz', 'Speed',
    'average_disturbed', 'average_quiet'
]

time_features = [
    'time_minutes', 'sin_hour', 'cos_hour', 'night'
]

physics_flags = [
    'high_kp', 'strong_storm', 'bz_neg',
    'night_high_kp', 'night_bz_neg'
]

physics_composites = [
    'Ey', 'storm_coupling'
]

candidate_features = storm_features + time_features + physics_flags + physics_composites
candidate_features = [c for c in candidate_features if c in df.columns]

# ensure numeric + drop constant features
for c in candidate_features:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

candidate_features = [c for c in candidate_features if df[c].nunique() > 1]

X_all = df[candidate_features].astype(float)
y_all = df[LABEL_COL].astype(int)
groups_all = df['date']  # group by date to avoid leakage

print("\nFeatures used:\n", candidate_features)
print("X_all shape:", X_all.shape, "| y_all shape:", y_all.shape)

# Quick correlation row with 'delay_binary'
corr_all = pd.concat([X_all, y_all.rename(LABEL_COL)], axis=1).corr(numeric_only=True)
if LABEL_COL in corr_all.index:
    delay_corr = corr_all.loc[LABEL_COL, X_all.columns].sort_values(key=np.abs, ascending=False)
    print("\nPearson correlation with target (sorted by |r|):")
    print(delay_corr.round(3).to_string())
    plt.figure(figsize=(min(10, 1.2 * len(X_all.columns)), 1.8))
    sns.heatmap(delay_corr.to_frame().T, annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-0.5, vmax=0.5, cbar=True)
    plt.title(f"Correlation of each feature with '{LABEL_COL}'")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# 6) Train/test split by date (to avoid leakage)
# -------------------------------------------------
unique_dates = pd.Series(df['date'].unique()).dropna()
train_dates, test_dates = train_test_split(
    unique_dates,
    test_size=0.2,
    random_state=RANDOM_STATE
)

train_mask = df['date'].isin(train_dates)
test_mask = df['date'].isin(test_dates)

X_train = X_all[train_mask]
X_test = X_all[test_mask]
y_train = y_all[train_mask]
y_test = y_all[test_mask]
groups_train = df.loc[train_mask, 'date']

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

gkf = GroupKFold(n_splits=5)

# -------------------------------------------------
# 7) Logistic regression with degree-2 interactions
# -------------------------------------------------
storm_subset = [c for c in storm_features if c in X_train.columns]
if not storm_subset:
    storm_subset = list(X_train.columns)

logit_pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scale", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(
        max_iter=4000,
        class_weight='balanced',
        solver='liblinear',
        random_state=RANDOM_STATE
    ))
])

param_grid_lr = {
    "poly__degree": [2],
    "clf__penalty": ["l1", "l2"],
    "clf__C": [0.1, 1.0, 5.0]
}

gs_lr = GridSearchCV(
    logit_pipeline,
    param_grid_lr,
    cv=gkf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

# For logistic with poly we can use all features
gs_lr.fit(X_train, y_train, groups=groups_train)
best_lr = gs_lr.best_estimator_
print("\nBest Logistic (poly) params:", gs_lr.best_params_)
print("Best Logistic (poly) CV AUC:", gs_lr.best_score_)
_ = evaluate_model("LogReg + degree-2 interactions", best_lr, X_test, y_test)


# -------------------------------------------------
# 8) Random Forest with CV + tuning
# -------------------------------------------------
rf_base = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

param_grid_rf = {
    "n_estimators": [200, 400, 800],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5]
}

rf_cv = GridSearchCV(
    rf_base,
    param_grid_rf,
    cv=gkf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

rf_cv.fit(X_train, y_train, groups=groups_train)
best_rf = rf_cv.best_estimator_
print("\nBest RF params:", rf_cv.best_params_)
print("Best RF CV AUC:", rf_cv.best_score_)
_ = evaluate_model("RandomForest (tuned)", best_rf, X_test, y_test)

perm_rf = permutation_importance(
    best_rf,
    X_test,
    y_test,
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
imp_rf = pd.DataFrame({
    "feature": X_train.columns,
    "importance": perm_rf.importances_mean
}).sort_values("importance", ascending=False)
print("\nPermutation importance (RF):\n", imp_rf.head(12).to_string(index=False))

plt.figure(figsize=(7, 4))
sns.barplot(data=imp_rf.head(12), x="importance", y="feature", orient="h")
plt.title("Permutation Importance — RandomForest (tuned)")
plt.tight_layout()
plt.show()


# -------------------------------------------------
# 9) XGBoost with CV + tuning
# -------------------------------------------------
try:
    from xgboost import XGBClassifier

    xgb_base = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    param_grid_xgb = {
        "n_estimators": [200, 400, 800],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "min_child_weight": [1, 5, 10]
    }

    xgb_cv = GridSearchCV(
        xgb_base,
        param_grid_xgb,
        cv=gkf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )

    xgb_cv.fit(X_train, y_train, groups=groups_train)
    best_xgb = xgb_cv.best_estimator_

    print("\nBest XGB params:", xgb_cv.best_params_)
    print("Best XGB CV AUC:", xgb_cv.best_score_)
    _ = evaluate_model("XGBoost (tuned)", best_xgb, X_test, y_test)

    perm_xgb = permutation_importance(
        best_xgb,
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    imp_xgb = pd.DataFrame({
        "feature": X_train.columns,
        "importance": perm_xgb.importances_mean
    }).sort_values("importance", ascending=False)
    print("\nPermutation importance (XGB):\n", imp_xgb.head(12).to_string(index=False))

    plt.figure(figsize=(7, 4))
    sns.barplot(data=imp_xgb.head(12), x="importance", y="feature", orient="h")
    plt.title("Permutation Importance — XGBoost (tuned)")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("\n[XGBoost not available or failed — skipping XGB tuning]")
    print(e)
