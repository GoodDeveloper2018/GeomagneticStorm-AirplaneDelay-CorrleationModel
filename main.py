import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# ======================================================================
# Integrating the logic from both models
# ======================================================================

# File paths (replace with your actual paths)
flight_data_path = 'flight_data_file_path'
geomagnetic_storm_data_path_excel = r'C:\Users\wangc\Downloads\Geomagnetic Storms - duration.xlsx'  # New model source
geomagnetic_storm_data_path_csv = 'geomagnetic_storm_data_file_path'  # Current model source

# ----------------------------------------------------------------------
# Load flight data and rename columns as in the current (original) model
# ----------------------------------------------------------------------
flight_data = pd.read_csv(flight_data_path)
flight_data.rename(columns={'Date': 'date', 'Time': 'time', 'Total Delay': 'delay'}, inplace=True)

# Drop unnecessary columns as in the current model
columns_to_drop = [
    'Terminal', 'Call Sign', 'Marketing Airline', 'General Aircraft Desc',
    'Max Takeoff Wt (Lbs)', 'Max Landing Wt (Lbs)', 'Intl / Dom',
    'Total Seats', 'Total Taxi Time', 'Direction', 'PA Airport', 'Non-PA Airport'
]
for col in columns_to_drop:
    if col in flight_data.columns:
        flight_data.drop(columns=[col], inplace=True)

# Ensure date is a proper datetime
flight_data['date'] = pd.to_datetime(flight_data['date'], errors='coerce')
flight_data.dropna(subset=['date'], inplace=True)

# ----------------------------------------------------------------------
# Load and process geomagnetic storm data using the new model approach
# ----------------------------------------------------------------------
geomagnetic_storm_data = pd.read_excel(geomagnetic_storm_data_path_excel)
geomagnetic_storm_data.columns = geomagnetic_storm_data.columns.str.strip()

# Create a date column by combining year and month (new model logic)
geomagnetic_storm_data['date'] = pd.to_datetime(
    geomagnetic_storm_data['YYYY'].astype(str) + geomagnetic_storm_data['MM'].astype(str).str.zfill(2),
    format='%Y%m'
)

# Compute average disturbed and quiet days (from the new model)
geomagnetic_storm_data['average_disturbed'] = geomagnetic_storm_data[['d1', 'd2', 'd3', 'd4', 'd5']].mean(axis=1)
geomagnetic_storm_data['average_quiet'] = geomagnetic_storm_data[['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q0']].mean(axis=1)

# Filter necessary columns and remove rows without proper dates
geomagnetic_storm_data = geomagnetic_storm_data.dropna(subset=['date'])

# ----------------------------------------------------------------------
# Merge the geomagnetic data with flight data (keeping useful features)
# ----------------------------------------------------------------------
data = pd.merge(flight_data, geomagnetic_storm_data[['date', 'average_disturbed', 'average_quiet']], on='date', how='inner')

# ----------------------------------------------------------------------
# Additional Geomagnetic Data (from current model CSV, if needed)
# NOTE: If the CSV data overlaps or provides extra columns like Dst, Ap, Kp max, Speed, IMF Bt, IMF Bz,
# we integrate them here. Otherwise, assume the new excel data already contained those columns.
# ----------------------------------------------------------------------
# Load the CSV-based geomagnetic data if it has unique indices/features
geomagnetic_storm_data_csv = pd.read_csv(geomagnetic_storm_data_path_csv)
geomagnetic_storm_data_csv.columns = geomagnetic_storm_data_csv.columns.str.strip()
geomagnetic_storm_data_csv['date'] = pd.to_datetime(geomagnetic_storm_data_csv['Date'], errors='coerce')
geomagnetic_storm_data_csv = geomagnetic_storm_data_csv.dropna(subset=['date'])

# Merge CSV-based geomagnetic data to supplement columns like Dst, Ap, Kp max, Speed, IMF Bt, IMF Bz
data = pd.merge(data, geomagnetic_storm_data_csv, on='date', how='inner')

# ----------------------------------------------------------------------
# Data Cleaning and Feature Engineering (current model logic)
# ----------------------------------------------------------------------
data = data.dropna()

# Convert delay to numeric and then to binary (delayed/not delayed)
data['delay'] = pd.to_numeric(data['delay'], errors='coerce')
data['delay'] = data['delay'].apply(lambda x: 1 if x > 0 else 0)

# Convert time to minutes since midnight
data['time'] = pd.to_datetime(data['time'], errors='coerce').dt.hour * 60 + pd.to_datetime(data['time'], errors='coerce').dt.minute

# Ensure all relevant columns are numeric; remove units if present
geomagnetic_columns = ['Dst', 'Kp max', 'Speed', 'IMF Bt', 'IMF Bz', 'Ap']
for column in geomagnetic_columns:
    if column in data.columns:
        # Remove known units (e.g. ' nT', ' Kp', 'km/sec') from current model
        data[column] = data[column].astype(str).str.replace(' nT', '', regex=False)
        data[column] = data[column].str.replace(' Kp', '', regex=False)
        data[column] = data[column].str.replace('km/sec', '', regex=False)
        data[column] = pd.to_numeric(data[column], errors='coerce')

# Drop rows missing essential features
all_required_features = ['time', 'Dst', 'Ap', 'Speed', 'IMF Bt', 'IMF Bz', 'average_disturbed', 'average_quiet']
data = data.dropna(subset=all_required_features + ['delay'])

# ----------------------------------------------------------------------
# Feature Selection (Integrate New Model Features + Current Model)
# We now have average_disturbed and average_quiet from the new model
# along with the geomagnetic indices and time from the current model
# ----------------------------------------------------------------------
features = ['time', 'Dst', 'Ap', 'Speed', 'IMF Bt', 'IMF Bz', 'average_disturbed', 'average_quiet']
X = data[features]
y = data['delay']

# ----------------------------------------------------------------------
# Train/Test Split, Scaling, and Modeling
# (Retain advanced interpretation from current model)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Updated correlation matrix with new features
corr_matrix = data[features + ['delay']].corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Updated Correlation Matrix with Geomagnetic Data')
plt.show()

# Train Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

# Predictions and confusion matrix
y_pred = logreg.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification report
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_text = classification_report(y_test, y_pred)

# ----------------------------------------------------------------------
# Retain the CorrelationOutcome Class and Conclusions from Current Model
# ----------------------------------------------------------------------
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
        # Using Dst index as a proxy for geomagnetic storm intensity correlation as per original logic
        if accuracy > 0.7:
            return "True: There is a significant correlation between geomagnetic storm intensity (Dst index) and flight delays."
        else:
            return "False: There is no significant correlation between geomagnetic storm intensity (Dst index) and flight delays."

# Evaluate correlation outcome
correlation_outcome = CorrelationOutcome(logreg)
accuracy, y_pred_thresh = correlation_outcome.evaluate(X_test_scaled, y_test)
conclusion = correlation_outcome.conclusive_sentence(accuracy)
print(conclusion)

# Visualization of model performance metrics
plt.figure(figsize=(10, 6))
sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1-Score'], y=[
    accuracy,
    class_report['1']['precision'],
    class_report['1']['recall'],
    class_report['1']['f1-score']
])
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()

# Summary of findings
classification_report_text = eval(classification_report)
summary_of_findings = f"""
Summary of Findings:
Cross-validation accuracy: {cv_mean:.2f} ± {cv_std:.2f}
Confusion Matrix:
{conf_matrix}
{classification_report_text}
"""
print(summary_of_findings)

# ----------------------------------------------------------------------
# Decision Boundary Plot (Retain from Current Model, but can select two geomagnetic features)
# ----------------------------------------------------------------------
def plot_decision_boundary(X, y, model, features):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Logistic Regression Decision Boundary')

# Select two features for decision boundary plot (e.g., 'average_disturbed' and 'average_quiet')
selected_features = ['average_disturbed', 'average_quiet']
X_selected = data[selected_features].values
y_selected = data['delay'].values

X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y_selected, test_size=0.2, random_state=42)

X_train_sel_scaled = scaler.fit_transform(X_train_sel)
X_test_sel_scaled = scaler.transform(X_test_sel)

logreg_sel = LogisticRegression(max_iter=1000)
logreg_sel.fit(X_train_sel_scaled, y_train_sel)

plt.figure(figsize=(10, 6))
plot_decision_boundary(X_test_sel_scaled, y_test_sel, logreg_sel, selected_features)
plt.show()
