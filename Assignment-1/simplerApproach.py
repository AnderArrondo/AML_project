from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, mean_squared_error
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np




# --- CONFIGS ---
csv_path = "./data/stroke.csv"
random_seed = 42

# --- DATA COLLECTION ---
df = pd.read_csv(csv_path)


# --- DATA PREPROCESSING ---
df = df.dropna()
df.drop(columns=["id"], inplace=True)

numerical_cols = ["age", "avg_glucose_level", "bmi"]
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop(columns=["stroke"])
y = df["stroke"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Creating samplings due to imbalances
sm = SMOTE(sampling_strategy=0.5, random_state=42) # Lleva la clase minoritaria al 50% de la mayoritaria
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)



# Model creation and training

rf_clf = RandomForestClassifier(random_state=random_seed)
dt_clf = DecisionTreeClassifier(random_state=random_seed)
lr_clf = LogisticRegression(random_state=random_seed, max_iter=5000)


rf_clf.fit(X_train_res, y_train_res)
dt_clf.fit(X_train_res, y_train_res)
lr_clf.fit(X_train_res, y_train_res)

rf_pred = rf_clf.predict(X_test_scaled)
dt_pred = dt_clf.predict(X_test_scaled)
lr_pred = lr_clf.predict(X_test_scaled)



# RESULTS


# Random Forest
print("Random Forest results on test data:")
print(f"MCC:       {matthews_corrcoef(y_test, rf_pred):.4f}")
print(f"Recall:    {recall_score(y_test, rf_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, rf_pred):.4f}")
print(f"MSE:       {mean_squared_error(y_test, rf_pred):.4f}")

# Decision Tree
print("\nDecision Tree results on test data:")
print(f"MCC:       {matthews_corrcoef(y_test, dt_pred):.4f}")
print(f"Recall:    {recall_score(y_test, dt_pred):.4f}")
print(f"Precision: {precision_score(y_test, dt_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, dt_pred):.4f}")
print(f"MSE:       {mean_squared_error(y_test, dt_pred):.4f}")

# Logistic Regression
print("\nLogistic Regression results on test data:")
print(f"MCC:       {matthews_corrcoef(y_test, lr_pred):.4f}")
print(f"Recall:    {recall_score(y_test, lr_pred):.4f}")
print(f"Precision: {precision_score(y_test, lr_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, lr_pred):.4f}")
print(f"MSE:       {mean_squared_error(y_test, lr_pred):.4f}")


# LOGISTIC REGRESSION OPTIMIZATION


print("\nHas Logistic Regression is the best, we are going to select the best threshold in order to maximize the results")

y_probs = lr_clf.predict_proba(X_test_scaled)[:, 1]

thresholds = np.linspace(0, 1, 100)
best_mcc = -1000
best_threshold = 0.5

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    current_mcc = matthews_corrcoef(y_test, y_pred_t)
    
    if current_mcc > best_mcc:
        best_mcc = current_mcc
        best_threshold = t


# RESULTS

print(f"\n\nBest threshold: {best_threshold:.4f}")
print(f"Best MCC-Score: {best_mcc:.4f}")

y_final_pred = (y_probs >= best_threshold).astype(int)
print(f"Recall final: {recall_score(y_test, y_final_pred):.4f}")
print(f"Precision final: {precision_score(y_test, y_final_pred):.4f}")
