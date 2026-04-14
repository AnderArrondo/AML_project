
import optuna
from optuna.samplers import TPESampler

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef
)

from xgboost import XGBClassifier

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler


# --- CONFIGS ---
csv_path = "./data/stroke.csv"
random_seed = 42
n_splits = 5

# --- DATA COLLECTION ---
df = pd.read_csv(csv_path)

# --- DATA PREPROCESSING ---
df.drop(columns=["id"], inplace=True)

numerical_cols = ["age", "avg_glucose_level", "bmi"]
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop(columns=["stroke"])
y = df["stroke"]

# X["age_glucose"] = X["age"] * X["avg_glucose_level"]
# X["bmi_age"] = X["bmi"] * X["age"]
# X["high_risk"] = ((X["age"] > 60) & (X["avg_glucose_level"] > 140)).astype(int)

smoking_cols = [c for c in X.columns if c.startswith("smoking_status_")]

# ---DATA SPLIT ---
X_cv, X_val, y_cv, y_val = train_test_split(
    X, y, test_size=0.2, random_state=random_seed, stratify=y
)

def objective(trial):
    # --- HYPERPARAMETER SEARCH SPACE ---
    # params = {
    #     "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    #     "max_depth": trial.suggest_int("max_depth", 5, 30),
    #     "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    #     "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    #     "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    #     "random_state": random_seed,
    #     "class_weight": "balanced"
    # }
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    threshold = trial.suggest_float("threshold", 0.01, 0.35)
    knn_n_neighs = trial.suggest_int("knn_n_neighs", 3, 10)
    sampling_strategy = trial.suggest_float("sampling_strategy", 0.1, 0.5)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    mccs = []

    for step, (train_idx, test_idx) in enumerate(skf.split(X_cv, y_cv)):
        X_train, X_test = X_cv.iloc[train_idx].copy(), X_cv.iloc[test_idx].copy()
        y_train, y_test = y_cv.iloc[train_idx], y_cv.iloc[test_idx]


        knn_imputer = KNNImputer(n_neighbors=knn_n_neighs)
        X_train = pd.DataFrame(knn_imputer.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(knn_imputer.transform(X_test), columns=X.columns)

        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        # rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_seed)
        # X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
        smotetomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_seed)
        X_train_res, y_train_res = smotetomek.fit_resample(X_train, y_train)

        # model = RandomForestClassifier(**params)
        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1
        )
        model.fit(X_train_res, y_train_res)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype(int)

        current_mcc = matthews_corrcoef(y_test, y_pred)
        mccs.append(current_mcc)

        intermediate_value = np.mean(mccs)
        trial.report(intermediate_value, step)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(mccs)

# --- RUN OPTIMIZATION ---
sampler = TPESampler(seed=random_seed)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=1000, n_jobs=2, show_progress_bar=True)

print("\n--- OPTUNA BEST PARAMS ---")
print(study.best_params)
print(f"Best Mean MCC: {study.best_value:.4f}")

best_params_dict = study.best_params.copy()
final_threshold = best_params_dict.pop('threshold')
knn_n_neighs = best_params_dict.pop('knn_n_neighs')
sampling_strategy = best_params_dict.pop('sampling_strategy')

# --- FINAL EVALUATION ON VALIDATION SET ---
knn_imputer = KNNImputer(n_neighbors=knn_n_neighs)
X_cv_imputed = pd.DataFrame(knn_imputer.fit_transform(X_cv), columns=X.columns)
X_val_imputed = pd.DataFrame(knn_imputer.transform(X_val), columns=X.columns)

scaler = StandardScaler()
X_cv_imputed[numerical_cols] = scaler.fit_transform(X_cv_imputed[numerical_cols])
X_val_imputed[numerical_cols] = scaler.transform(X_val_imputed[numerical_cols])

# rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_seed)
# X_res, y_res = rus.fit_resample(X_cv_imputed, y_cv)
smotetomek = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_seed)
X_res, y_res = smotetomek.fit_resample(X_cv_imputed, y_cv)

best_model = XGBClassifier(
    **best_params_dict,
    eval_metric="logloss",
    random_state=random_seed,
    n_jobs=-1
)
best_model.fit(X_res, y_res)

y_val_pred = (best_model.predict_proba(X_val_imputed)[:, 1] >= final_threshold).astype(int)
print("\n--- FINAL VALIDATION MCC ---")
print(matthews_corrcoef(y_val, y_val_pred))

with open("1000trials.txt", "w") as f:
    f.write(str(best_params_dict))
    f.write(matthews_corrcoef(y_val, y_val_pred))