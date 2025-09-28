# =========================================
# End-to-end: prep + engineered flags + models
# =========================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# -----------------------
# 1) LOAD / CLEAN / MERGE
# -----------------------
flight_data_path = r"C:\Users\arshp\OneDrive\Desktop\Coding Work\Terra Research\JFK-LAX flights 2023 Model.csv"
geomagnetic_storm_duration_data_path = r"C:\Users\arshp\OneDrive\Desktop\Coding Work\Terra Research\Geomagnetic Storms - duration.xlsx"
geomagnetic_storm_data_points_path = r"C:\Users\arshp\OneDrive\Desktop\Coding Work\Terra Research\Geomagnetic-Storm_Data.csv"

def clean_cols(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[()]", "", regex=True)
        .str.replace("__+", "_", regex=True)
    )
    return df

# Load
flight = pd.read_csv(flight_data_path)
storm_monthly = pd.read_excel(geomagnetic_storm_duration_data_path)
storm_daily = pd.read_csv(geomagnetic_storm_data_points_path)

# Normalize column names
flight = clean_cols(flight)
storm_monthly = clean_cols(storm_monthly)
storm_daily = clean_cols(storm_daily)

# Flights
if {'Date','Time','Total_Delay'}.issubset(set(flight.columns)):
    flight = flight.rename(columns={'Date':'date','Time':'time','Total_Delay':'delay'})
elif {'date','time','total_delay'}.issubset(set(flight.columns)):
    flight = flight.rename(columns={'total_delay':'delay'})

flight['date'] = pd.to_datetime(flight['date'], errors='coerce').dt.date
drop_cols = ['Terminal','Call_Sign','Marketing_Airline','General_Aircraft_Desc',
             'Max_Takeoff_Wt_Lbs','Max_Landing_Wt_Lbs','Intl/_Dom','Total_Seats',
             'Total_Taxi_Time','Direction','PA_Airport','Non-PA_Airport']
flight = flight.drop(columns=[c for c in drop_cols if c in flight.columns], errors='ignore')
print("flight rows (with valid date):", flight['date'].notna().sum())

# Monthly duration → month key + aggregates
if {'YYYY','MM'}.issubset(set(storm_monthly.columns)):
    storm_monthly['month'] = pd.to_datetime(
        storm_monthly['YYYY'].astype(int).astype(str) + '-' +
        storm_monthly['MM'].astype(int).astype(str).str.zfill(2) + '-01'
    ).dt.to_period('M')
elif 'date' in storm_monthly.columns:
    storm_monthly['month'] = pd.to_datetime(storm_monthly['date'], errors='coerce').dt.to_period('M')
else:
    raise ValueError("Monthly table needs YYYY/MM or date.")

disturbed_cols = [c for c in storm_monthly.columns if c.lower().startswith('d')]
quiet_cols     = [c for c in storm_monthly.columns if c.lower().startswith('q')]

if disturbed_cols:
    vals_d = storm_monthly[disturbed_cols].apply(pd.to_numeric, errors='coerce')
    has_any = vals_d.notna().sum(axis=1) > 0
    storm_monthly['average_disturbed'] = np.where(has_any, vals_d.mean(axis=1), np.nan)

if quiet_cols:
    vals_q = storm_monthly[quiet_cols].apply(pd.to_numeric, errors='coerce')
    has_any = vals_q.notna().sum(axis=1) > 0
    storm_monthly['average_quiet'] = np.where(has_any, vals_q.mean(axis=1), np.nan)

storm_monthly = storm_monthly[['month','average_disturbed','average_quiet']].drop_duplicates()
print("storm_monthly months:", storm_monthly['month'].nunique())

# Daily storm metrics — robust parsing
# map column name variants
name_map = {}
for col in list(storm_daily.columns):
    c = col.lower().replace(" ", "_").replace("-", "_")
    if "kp" in c and "max" in c: name_map[col] = "Kp_max"
    elif c in ("kp", "k_p"):     name_map[col] = "Kp_max"
    elif c.startswith("ap"):     name_map[col] = "Ap"
    elif c.startswith("dst"):    name_map[col] = "Dst"
    elif "imf" in c and "bt" in c: name_map[col] = "IMF_Bt"
    elif "imf" in c and "bz" in c: name_map[col] = "IMF_Bz"
    elif "speed" in c or "km_s" in c: name_map[col] = "Speed"
storm_daily = storm_daily.rename(columns=name_map)

if 'date' in storm_daily.columns:
    storm_daily['date'] = pd.to_datetime(storm_daily['date'], errors='coerce').dt.date
elif 'Date' in storm_daily.columns:
    storm_daily['date'] = pd.to_datetime(storm_daily['Date'], errors='coerce').dt.date
else:
    raise ValueError("Daily table missing date/Date column.")

import re
def first_number(s):
    if pd.isna(s): return np.nan
    s = str(s).replace("\u2212","-")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def last_number(s):
    if pd.isna(s): return np.nan
    s = str(s).replace("\u2212","-")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else np.nan

for col in ['Dst','Ap','Kp_max','Speed','IMF_Bt','IMF_Bz']:
    if col in storm_daily.columns:
        if col == 'Kp_max' or col == 'Speed':
            storm_daily[col] = storm_daily[col].map(first_number)
        else:
            storm_daily[col] = storm_daily[col].map(last_number)

keep_cols = ['date','Dst','Ap','Kp_max','IMF_Bt','IMF_Bz','Speed']
storm_daily = storm_daily[[c for c in keep_cols if c in storm_daily.columns]].dropna(subset=['date'])
print("storm_daily rows (with date):", len(storm_daily))
print("storm_daily coverage:\n", storm_daily.notna().mean().round(3))

# Merge day-by-day
df = pd.merge(flight, storm_daily, on='date', how='inner')
print("after flights ⨉ dailystorm:", df.shape)

# Add monthly aggregates
df['month'] = pd.to_datetime(df['date'], errors='coerce').dt.to_period('M')
df = pd.merge(df, storm_monthly, on='month', how='left')
print("after + monthly:", df.shape)

# Label + NAs
df['delay'] = pd.to_numeric(df['delay'], errors='coerce')
df = df.dropna(subset=['delay'])
df['delay'] = (df['delay'] > 0).astype(int)

for c in ['Dst','Ap','Kp_max','IMF_Bt','IMF_Bz','Speed','average_disturbed','average_quiet']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

def parse_time_series(s):
    s = s.astype(str).str.strip()
    t = pd.to_datetime(s, format='%H:%M', errors='coerce')
    m = t.isna()
    t.loc[m] = pd.to_datetime(s[m], format='%I:%M %p', errors='coerce')
    m = t.isna()
    t.loc[m] = pd.to_datetime(s[m], format='%H:%M:%S', errors='coerce')
    return t

t1 = parse_time_series(df['time'])
df['time_minutes'] = t1.dt.hour.fillna(0)*60 + t1.dt.minute.fillna(0)
df['hour'] = t1.dt.hour.fillna(0)
df['sin_hour'] = np.sin(2*np.pi*df['hour']/24.0)
df['cos_hour'] = np.cos(2*np.pi*df['hour']/24.0)

print("final rows ready:", len(df))
print("delay distribution:\n", df['delay'].value_counts(dropna=False))

# ---------------------------------------------
# 2) EXTRA, interpretable engineered indicators
# ---------------------------------------------
if 'Kp_max' in df: df['high_kp'] = (df['Kp_max'] >= 6).astype(int)
if 'Dst' in df:    df['strong_storm'] = (df['Dst'] <= -70).astype(int)
if 'IMF_Bz' in df: df['bz_neg'] = (df['IMF_Bz'] <= -10).astype(int)
df['night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

# Feature set
# =========================
# BUILD X and y (cleanly)
# =========================
# 1) Choose candidates that should exist if upstream cleaning ran.
candidate_list = [
    'Dst','Ap','Kp_max','IMF_Bt','IMF_Bz','Speed',
    'average_disturbed','average_quiet',
    'time_minutes','sin_hour','cos_hour'
]
present = [c for c in candidate_list if c in df.columns]

# 2) Force numeric + fill remaining NaNs (robust, won’t drop rows)
for c in present:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# 3) Drop constant/no-variance columns (they break correlations)
present = [c for c in present if df[c].nunique(dropna=True) > 1]

# 4) Build X (features) and y (label)
X = df[present].astype(float)
y = df['delay'].astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Feature coverage (non-null %):\n", X.notna().mean().round(3))

# =========================
# Correlation with 'delay'
# =========================
# Pearson (point-biserial for binary y)
corr_full = pd.concat([X, y.rename('delay')], axis=1).corr(numeric_only=True)
pear_delay = corr_full.loc['delay', X.columns].sort_values(key=np.abs, ascending=False)

print("\nPearson correlation with delay (sorted by |r|):")
print(pear_delay.round(3).to_string())

# Spearman & Kendall (rank-based)
from scipy.stats import spearmanr, kendalltau
import numpy as np

spearman_vals = {c: spearmanr(X[c], y).correlation for c in X.columns}
kendall_vals  = {c: kendalltau(X[c], y).correlation  for c in X.columns}

spearman_series = pd.Series(spearman_vals).sort_values(key=np.abs, ascending=False)
kendall_series  = pd.Series(kendall_vals).sort_values(key=np.abs, ascending=False)

print("\nSpearman correlation with delay:")
print(spearman_series.round(3).to_string())
print("\nKendall correlation with delay:")
print(kendall_series.round(3).to_string())

# Mini heatmap: only the 'delay' row (no 1.0 self-corr)
delay_row = corr_full.loc[['delay'], X.columns]

plt.figure(figsize=(min(10, 1.2*len(X.columns)), 1.8))
sns.heatmap(delay_row, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.5, vmax=0.5, cbar=True)
plt.title("Correlation of each feature with 'delay'")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

present = [c for c in candidate_list if c in df.columns]
coverage = df[present].notna().mean()
low = coverage[coverage < 0.20].index.tolist()
if low:
    print("Dropping low-coverage features:", low)
    df = df.drop(columns=low)

feature_candidates = [c for c in candidate_list if c in df.columns]
X = df[feature_candidates].astype(float)
y = df['delay'].astype(int)

# Quick correlation map (won’t capture non-linear)
non_const = X.loc[:, X.nunique() > 1]
corr_df = pd.concat([non_const, y.rename('delay')], axis=1)
if corr_df.shape[1] > 1:
    corr_matrix = corr_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Matrix')
    plt.show()

# ----------------------------------------------
# 3) MODEL SUITE — interactions & non-linearities
# ----------------------------------------------
rng = 42
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rng, stratify=y
)

def evaluate(model_name, fitted_model, Xt, yt, prob_method="predict_proba"):
    if prob_method == "predict_proba":
        y_prob = fitted_model.predict_proba(Xt)[:, 1]
    else:
        y_prob = fitted_model.decision_function(Xt)
        y_prob = (y_prob - y_prob.min())/(y_prob.max() - y_prob.min() + 1e-9)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(yt, y_prob)
    ap = average_precision_score(yt, y_prob)
    print(f"\n=== {model_name} ===")
    print(classification_report(yt, y_pred, digits=3))
    print(f"ROC-AUC: {auc:.3f} | PR-AUC: {ap:.3f}")
    cm = confusion_matrix(yt, y_pred)
    plt.figure(figsize=(4.6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix — {model_name}')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout(); plt.show()
    RocCurveDisplay.from_predictions(yt, y_prob)
    plt.title(f'ROC Curve — {model_name}'); plt.tight_layout(); plt.show()
    PrecisionRecallDisplay.from_predictions(yt, y_prob)
    plt.title(f'Precision–Recall — {model_name}'); plt.tight_layout(); plt.show()
    return auc, ap

# A) Single-feature screens (pure effects)
print("\n[ Single-feature screens (5-fold ROC-AUC / PR-AUC) ]")
single_results = []
for col in X.columns:
    Xi = X[[col]].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', random_state=rng))
    ])
    auc = cross_val_score(pipe, Xi, y, cv=cv, scoring="roc_auc").mean()
    ap  = cross_val_score(pipe, Xi, y, cv=cv, scoring="average_precision").mean()
    single_results.append((col, auc, ap))

single_df = pd.DataFrame(single_results, columns=["feature","cv_roc_auc","cv_pr_auc"])\
             .sort_values("cv_roc_auc", ascending=False)
print(single_df.to_string(index=False))
plt.figure(figsize=(8,4))
sns.barplot(data=single_df, x="cv_roc_auc", y="feature", orient="h")
plt.title("Single-feature ROC-AUC (5-fold)"); plt.xlim(0.4, 0.7); plt.tight_layout(); plt.show()

# B) Logistic with degree-2 interactions (on storm metrics only)
storm_feats = [c for c in ["Ap","Kp_max","Dst","IMF_Bt","IMF_Bz","Speed",
                           "average_disturbed","average_quiet"] if c in X.columns]
poly_cols = storm_feats if len(storm_feats) > 0 else X.columns.tolist()

logit_poly = Pipeline([
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("scale",  StandardScaler(with_mean=False)),
    ("clf",    LogisticRegression(max_iter=4000, class_weight='balanced', solver='liblinear'))
])
param_grid_lr = {
    "poly__degree": [2],
    "clf__penalty": ["l1","l2"],
    "clf__C": [0.1, 1.0, 5.0]
}
gs_lr = GridSearchCV(logit_poly, param_grid_lr, cv=cv, scoring="roc_auc", n_jobs=-1)

X_train_poly = pd.concat([X_train[poly_cols], X_train.drop(columns=poly_cols)], axis=1)
X_test_poly  = pd.concat([X_test[poly_cols],  X_test.drop(columns=poly_cols)],  axis=1)
gs_lr.fit(X_train_poly, y_train)
best_lr = gs_lr.best_estimator_
print("\nBest Logistic (poly) params:", gs_lr.best_params_)
_ = evaluate("LogReg + degree-2 interactions", best_lr, X_test_poly, y_test)

# C) Random Forest (non-linear, interactions implicit)
rf = RandomForestClassifier(
    n_estimators=600, max_depth=None, min_samples_split=4, min_samples_leaf=2,
    class_weight="balanced", random_state=rng, n_jobs=-1
)
rf.fit(X_train, y_train)
_ = evaluate("RandomForest", rf, X_test, y_test, prob_method="predict_proba")

# Permutation importance for RF
perm = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=rng, n_jobs=-1)
imp_rf = pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean})\
         .sort_values("importance", ascending=False)
print("\nPermutation importance (RF):\n", imp_rf.head(12).to_string(index=False))
plt.figure(figsize=(7,4))
sns.barplot(data=imp_rf.head(12), x="importance", y="feature", orient="h")
plt.title("Permutation Importance — RandomForest"); plt.tight_layout(); plt.show()

# D) XGBoost (optional)
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, eval_metric='logloss', random_state=rng
    )
    xgb.fit(X_train, y_train)
    _ = evaluate("XGBoost", xgb, X_test, y_test, prob_method="predict_proba")

    perm_x = permutation_importance(xgb, X_test, y_test, n_repeats=30, random_state=rng, n_jobs=-1)
    imp_x = pd.DataFrame({"feature": X.columns, "importance": perm_x.importances_mean})\
            .sort_values("importance", ascending=False)
    print("\nPermutation importance (XGB):\n", imp_x.head(12).to_string(index=False))
    plt.figure(figsize=(7,4))
    sns.barplot(data=imp_x.head(12), x="importance", y="feature", orient="h")
    plt.title("Permutation Importance — XGBoost"); plt.tight_layout(); plt.show()
except Exception:
    print("\n[XGBoost not installed — skipping XGB run]")
