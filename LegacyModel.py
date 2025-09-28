import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr, kendalltau

# ======================================================================
# Existing Code as Provided
# ======================================================================
flight_data_path = r"C:\Users\arshp\OneDrive\Desktop\Terra Research\JFK-LAX flights 2023 Model.csv"
geomagnetic_storm_duration_data_path = r"C:\Users\arshp\OneDrive\Desktop\Terra Research\Geomagnetic Storms - duration.xlsx"
geomagnetic_storm_data_points_path = r"C:\Users\arshp\OneDrive\Desktop\Terra Research\Geomagnetic-Storm_Data.csv"

flight_data = pd.read_csv(flight_data_path)
flight_data.rename(columns={'Date': 'date', 'Time': 'time', 'Total Delay': 'delay'}, inplace=True)

columns_to_drop = [
    'Terminal', 'Call Sign', 'Marketing Airline', 'General Aircraft Desc',
    'Max Takeoff Wt (Lbs)', 'Max Landing Wt (Lbs)', 'Intl / Dom',
    'Total Seats', 'Total Taxi Time', 'Direction', 'PA Airport', 'Non-PA Airport'
]
for col in columns_to_drop:
    if col in flight_data.columns:
        flight_data.drop(columns=[col], inplace=True)

flight_data['date'] = pd.to_datetime(flight_data['date'], errors='coerce')
flight_data.dropna(subset=['date'], inplace=True)

storm_duration_data = pd.read_excel(geomagnetic_storm_duration_data_path)
if all(c in storm_duration_data.columns for c in ['YYYY', 'MM']):
    # Constructing the date from only year and month, defaulting DD to 01
    storm_duration_data['date'] = pd.to_datetime(
        storm_duration_data['YYYY'].astype(str) + '-' +
        storm_duration_data['MM'].astype(str).str.zfill(2) + '-01',
        format='%Y-%m-%d'
    )
elif 'date' in storm_duration_data.columns:
    storm_duration_data['date'] = pd.to_datetime(storm_duration_data['date'], errors='coerce')
else:
    raise ValueError("No suitable date information found in storm_duration_data.")

storm_duration_data.dropna(subset=['date'], inplace=True)

geomagnetic_storm_points = pd.read_csv(geomagnetic_storm_data_points_path)
if 'Date' in geomagnetic_storm_points.columns:
    geomagnetic_storm_points['date'] = pd.to_datetime(geomagnetic_storm_points['Date'], errors='coerce')
    geomagnetic_storm_points.drop(columns=['Date'], inplace=True)
else:
    geomagnetic_storm_points['date'] = pd.to_datetime(geomagnetic_storm_points['date'], errors='coerce')
geomagnetic_storm_points.dropna(subset=['date'], inplace=True)

for c in geomagnetic_storm_points.columns:
    if c not in ['date']:
        geomagnetic_storm_points[c] = geomagnetic_storm_points[c].astype(str).str.replace(' nT', '', regex=False)
        geomagnetic_storm_points[c] = geomagnetic_storm_points[c].str.replace(' Kp', '', regex=False)
        geomagnetic_storm_points[c] = geomagnetic_storm_points[c].str.replace('km/sec', '', regex=False)
        geomagnetic_storm_points[c] = pd.to_numeric(geomagnetic_storm_points[c], errors='coerce')

geomagnetic_storm_points.dropna(subset=geomagnetic_storm_points.columns.difference(['date']), inplace=True)

geomagnetic_data_merged = pd.merge(storm_duration_data, geomagnetic_storm_points, on='date', how='inner')

disturbed_cols = [col for col in storm_duration_data.columns if 'd' in col]
quiet_cols = [col for col in storm_duration_data.columns if 'q' in col]
if disturbed_cols:
    geomagnetic_data_merged['average_disturbed'] = geomagnetic_data_merged[disturbed_cols].mean(axis=1)
if quiet_cols:
    geomagnetic_data_merged['average_quiet'] = geomagnetic_data_merged[quiet_cols].mean(axis=1)

geomagnetic_data_merged.dropna(subset=['date'], inplace=True)

data = pd.merge(flight_data, geomagnetic_data_merged, on='date', how='inner')
data.dropna(inplace=True)

data['delay'] = pd.to_numeric(data['delay'], errors='coerce').apply(lambda x: 1 if x > 0 else 0)
data['time'] = pd.to_datetime(data['time'], errors='coerce').dt.hour * 60 + pd.to_datetime(data['time'],
                                                                                           errors='coerce').dt.minute

feature_candidates = data.select_dtypes(include=[np.number]).columns.tolist()
feature_candidates = [c for c in feature_candidates if c not in ['delay']]

X = data[feature_candidates]
y = data['delay']

# ======================================================================
# Attempt to Boost Correlation Discovery
# ======================================================================

# Let's focus on known geomagnetic indices commonly related to storms:
indices_of_interest = ['Dst', 'Ap', 'Kp_max']
available_indices = [idx for idx in indices_of_interest if idx in X.columns]

print("Indices available:", available_indices)

# 1) Check Pearson correlation again
corr_matrix = data[feature_candidates + ['delay']].corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Matrix')
plt.show()

print("Pearson Correlations with delay:")
for idx in available_indices:
    pear_corr = corr_matrix.loc['delay', idx]
    print(f"{idx} vs delay (Pearson): {pear_corr:.3f}")

# 2) Spearman and Kendall correlation
print("\nSpearman and Kendall correlation with delay:")
for idx in available_indices:
    spearman_corr, _ = spearmanr(data[idx], data['delay'])
    kendall_corr, _ = kendalltau(data[idx], data['delay'])
    print(f"{idx} vs delay (Spearman): {spearman_corr:.3f}, (Kendall): {kendall_corr:.3f}")

# 3) Try transformations: absolute values or focusing on days with strong storms
# For example, filter to days where |Dst| > 50 nT or Ap > 30, etc., and see if correlation changes
transformed_data = data.copy()
if 'Dst' in transformed_data.columns:
    transformed_data['Dst_abs'] = transformed_data['Dst'].abs()

if 'Ap' in transformed_data.columns:
    transformed_data['Ap_log'] = np.log1p(transformed_data['Ap'].clip(lower=0))

if 'Kp_max' in transformed_data.columns:
    transformed_data['Kp_max_squared'] = transformed_data['Kp_max'] ** 2

new_features = ['Dst_abs', 'Ap_log', 'Kp_max_squared']
new_features = [f for f in new_features if f in transformed_data.columns]

if new_features:
    print("\nCorrelation after transformations:")
    transformed_corr = transformed_data[new_features + ['delay']].corr(numeric_only=True)
    print(transformed_corr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(transformed_corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix with Transformed Features')
    plt.show()

# 4) Focus on subset of data (e.g., where Dst < -50 or Ap > 30)
# Maybe the correlation only emerges under strong storm conditions.
subset_conditions = []
if 'Dst' in transformed_data.columns:
    subset_conditions.append(("Dst < -50", transformed_data[transformed_data['Dst'] < -50]))
if 'Ap' in transformed_data.columns:
    subset_conditions.append(("Ap > 30", transformed_data[transformed_data['Ap'] > 30]))
if 'Kp_max' in transformed_data.columns:
    subset_conditions.append(("Kp_max > 5", transformed_data[transformed_data['Kp_max'] > 5]))

for condition_str, subset in subset_conditions:
    if len(subset) > 10:  # Ensure there's enough data
        print(f"\nCorrelation under condition: {condition_str}")
        scorr = subset[available_indices + ['delay']].corr(numeric_only=True)
        print(scorr)
        plt.figure(figsize=(6, 5))
        sns.heatmap(scorr, annot=True, cmap='coolwarm')
        plt.title(f'Correlation under {condition_str}')
        plt.show()

# ======================================================================
# Re-run Logistic Model with transformed features (if any)
# ======================================================================
all_features = feature_candidates + new_features
X_transformed = transformed_data[all_features]
X_transformed = X_transformed.select_dtypes(include=[np.number]).fillna(0)
y_transformed = transformed_data['delay']

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix with Transformed Features')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

class_report_text = classification_report(y_test, y_pred)
print(class_report_text)


# Check if model got better accuracy
class CorrelationOutcome:
    def __init__(self, logistic_model, threshold=0.5):
        self.model = logistic_model
        self.threshold = threshold

    def evaluate(self, X, y):
        y_pred_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_prob >= self.threshold).astype(int)
        accuracy = (y_pred == y).mean()
        return accuracy, y_pred

    def conclusive_sentence(self, accuracy):
        if accuracy > 0.7:
            return "True: There is a significant correlation."
        else:
            return "False: Still no significant correlation."


correlation_outcome = CorrelationOutcome(logreg)
accuracy, _ = correlation_outcome.evaluate(X_test_scaled, y_test)
print("Model Accuracy with Transformed Features:", accuracy)
print(correlation_outcome.conclusive_sentence(accuracy))
