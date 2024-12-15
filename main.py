import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# ======================================================================
# File Paths (Replace with Actual Paths)
# ======================================================================
flight_data_path = r"C:\Users\arshp\OneDrive\Desktop\JFK-LAX flights 2023 Model.csv"
geomagnetic_storm_duration_data_path = r"C:\Users\arshp\OneDrive\Desktop\Geomagnetic Storms - duration.xlsx"
geomagnetic_storm_data_points_path = r"C:\Users\arshp\OneDrive\Desktop\geomagnetic_storm_data_points.csv"

# ======================================================================
# Load Flight Data
# ======================================================================
flight_data = pd.read_csv(flight_data_path)
flight_data.rename(columns={'Date': 'date', 'Time': 'time', 'Total Delay': 'delay'}, inplace=True)

# Drop unnecessary columns
columns_to_drop = [
    'Terminal', 'Call Sign', 'Marketing Airline', 'General Aircraft Desc',
    'Max Takeoff Wt (Lbs)', 'Max Landing Wt (Lbs)', 'Intl / Dom',
    'Total Seats', 'Total Taxi Time', 'Direction', 'PA Airport', 'Non-PA Airport'
]
for col in columns_to_drop:
    if col in flight_data.columns:
        flight_data.drop(columns=[col], inplace=True)

# Ensure date is datetime
flight_data['date'] = pd.to_datetime(flight_data['date'], errors='coerce')
flight_data.dropna(subset=['date'], inplace=True)

# ======================================================================
# Load and Process Storm Duration Data
# ======================================================================
storm_duration_data = pd.read_excel(geomagnetic_storm_duration_data_path)
# Make sure storm_duration_data has a 'date' column or create one if it has separate year/month/day columns.
# For demonstration, assume there's a 'date' column already or a way to form it.
# If the file has 'YYYY', 'MM', 'DD', we can do:
if all(c in storm_duration_data.columns for c in ['YYYY', 'MM', 'DD']):
    storm_duration_data['date'] = pd.to_datetime(
        storm_duration_data['YYYY'].astype(str) + '-' + 
        storm_duration_data['MM'].astype(str).str.zfill(2) + '-' + 
        storm_duration_data['DD'].astype(str).str.zfill(2),
        format='%Y-%m-%d'
    )
elif 'date' in storm_duration_data.columns:
    storm_duration_data['date'] = pd.to_datetime(storm_duration_data['date'], errors='coerce')
else:
    raise ValueError("No suitable date information found in storm_duration_data.")

storm_duration_data.dropna(subset=['date'], inplace=True)

# The storm_duration_data should contain features like storm durations or related metrics.
# Ensure these are numeric and cleaned if needed. For example:
numeric_cols = storm_duration_data.select_dtypes(include=[np.number]).columns.tolist()

# ======================================================================
# Load and Process Geomagnetic Storm Data Points
# ======================================================================
geomagnetic_storm_points = pd.read_csv(geomagnetic_storm_data_points_path)
# Ensure a proper date column
if 'Date' in geomagnetic_storm_points.columns:
    geomagnetic_storm_points['date'] = pd.to_datetime(geomagnetic_storm_points['Date'], errors='coerce')
    geomagnetic_storm_points.drop(columns=['Date'], inplace=True)
else:
    # If already has 'date', just ensure datetime
    geomagnetic_storm_points['date'] = pd.to_datetime(geomagnetic_storm_points['date'], errors='coerce')
geomagnetic_storm_points.dropna(subset=['date'], inplace=True)

# Clean geomagnetic features (remove units, etc.)
for c in geomagnetic_storm_points.columns:
    if c not in ['date']:
        # Remove common units if present
        geomagnetic_storm_points[c] = geomagnetic_storm_points[c].astype(str).str.replace(' nT', '', regex=False)
        geomagnetic_storm_points[c] = geomagnetic_storm_points[c].str.replace(' Kp', '', regex=False)
        geomagnetic_storm_points[c] = geomagnetic_storm_points[c].str.replace('km/sec', '', regex=False)
        geomagnetic_storm_points[c] = pd.to_numeric(geomagnetic_storm_points[c], errors='coerce')

geomagnetic_storm_points.dropna(subset=geomagnetic_storm_points.columns.difference(['date']), inplace=True)

# ======================================================================
# Merge Storm Duration and Storm Points Data
# ======================================================================
# Merge on 'date' to combine storm duration metrics with actual storm data points
geomagnetic_data_merged = pd.merge(storm_duration_data, geomagnetic_storm_points, on='date', how='inner')

# If you have columns like 'average_disturbed' and 'average_quiet', you can compute them if needed:
# For example, if storm_duration_data had disturbed/quiet day columns:
disturbed_cols = [col for col in storm_duration_data.columns if 'd' in col]
quiet_cols = [col for col in storm_duration_data.columns if 'q' in col]
if disturbed_cols:
    geomagnetic_data_merged['average_disturbed'] = geomagnetic_data_merged[disturbed_cols].mean(axis=1)
if quiet_cols:
    geomagnetic_data_merged['average_quiet'] = geomagnetic_data_merged[quiet_cols].mean(axis=1)

# Drop rows without proper merged data
geomagnetic_data_merged.dropna(subset=['date'], inplace=True)

# ======================================================================
# Merge Flight Data with Combined Geomagnetic Data
# ======================================================================
data = pd.merge(flight_data, geomagnetic_data_merged, on='date', how='inner')
data.dropna(inplace=True)

# Convert delay to binary
data['delay'] = pd.to_numeric(data['delay'], errors='coerce').apply(lambda x: 1 if x > 0 else 0)

# Convert time to minutes since midnight if it's a time column
data['time'] = pd.to_datetime(data['time'], errors='coerce').dt.hour * 60 + pd.to_datetime(data['time'], errors='coerce').dt.minute

# Identify potential feature columns (geomagnetic indices + storm duration features + time)
# Exclude non-numeric columns
feature_candidates = data.select_dtypes(include=[np.number]).columns.tolist()
feature_candidates = [c for c in feature_candidates if c not in ['delay']]  # exclude target

X = data[feature_candidates]
y = data['delay']

# ======================================================================
# Correlation Analysis
# ======================================================================
corr_matrix = data[feature_candidates + ['delay']].corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Merged Data')
plt.show()

# If correlation is low, consider feature selection, or try different aggregations/time windows.
# For modeling:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

y_pred = logreg.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

class_report_text = classification_report(y_test, y_pred)
print(class_report_text)

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
            return "True: There is a significant correlation between the chosen geomagnetic metrics and flight delays."
        else:
            return "False: There is no significant correlation between the chosen geomagnetic metrics and flight delays."

correlation_outcome = CorrelationOutcome(logreg)
accuracy, y_pred_thresh = correlation_outcome.evaluate(X_test_scaled, y_test)
conclusion = correlation_outcome.conclusive_sentence(accuracy)
print(conclusion)

# Display model performance
report_dict = classification_report(y_test, y_pred, output_dict=True)
plt.figure(figsize=(10, 6))
sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
            y=[accuracy,
               report_dict['1']['precision'],
               report_dict['1']['recall'],
               report_dict['1']['f1-score']])
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()

summary_of_findings = f"""
Summary of Findings:
Cross-validation accuracy: {cv_mean:.2f} ± {cv_std:.2f}
Confusion Matrix:
{conf_matrix}
{class_report_text}
"""
print(summary_of_findings)
