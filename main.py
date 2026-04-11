
import optuna
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, SVMSMOTE, BorderlineSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score


# Imbalanced dataset
## Class weights
## Oversampling
### Random
#### Sampling strategies
### SMOTE
### ADASYN
### stratification
## Undersampling

# DATA COLLECTION
random_seed = 42
csv_path = "./data/stroke.csv"
df=pd.read_csv(csv_path)

# DATA PREPROCESSING
df=df.drop(columns=["id"])
df["age"]=df["age"].apply(int)
df["ever_married"] = df["ever_married"].map({"Yes":1, "No":0})
df=df.rename(columns={"Residence_type":"is_rural"})
df["is_rural"]=df["is_rural"].map({"Rural":1,"Urban":0})
df_dummies= pd.get_dummies(df,columns=["work_type","gender","smoking_status"])

print(df.head())
print(df_dummies.head())

X = df_dummies.drop(columns=['stroke'])
y = df_dummies['stroke']
X_np = X.values.astype(np.float32)
y_np = y.values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
splits = list(skf.split(X, y))

def objective(trial):
    # imputation
    imputer_type = trial.suggest_categorical("imputer_type", ["simple", "knn"])
    if imputer_type == "simple":
        strat = trial.suggest_categorical("simp_imp_strategy", ["mean", "median", "most_frequent"])
        imputer = SimpleImputer(strategy=strat)
    elif imputer_type == "knn":
        n_neighbors = trial.suggest_int("knn_input_neighs", 3, 20)
        imputer = KNNImputer(n_neighbors=n_neighbors)

    # sampling
    sampler_type = trial.suggest_categorical(
        "sampler_type",
        ["random_over", "random_under", "adasyn", "smote", "none"]
    
    )
    if sampler_type == "random_over":
        sampler = RandomOverSampler(random_state=random_seed)
    elif sampler_type == "random_under":
        sampler = RandomUnderSampler(random_state=random_seed)
    elif sampler_type == "adasyn":
        n_adasyn = trial.suggest_int("adasyn_n_neighs", 3, 20)
        sampler = ADASYN(n_neighbors=n_adasyn, random_state=random_seed)
    elif sampler_type == "smote":
        n_smote = trial.suggest_int("smote_n_neighs", 3, 20)
        sampler = SMOTE(k_neighbors=n_smote, random_state=random_seed)
    else:
        sampler = "passthrough"

    # classifier
    classifier_type = trial.suggest_categorical(
        "classifier_type",
        ["rf", "xgb", "adab"]
    )
    if classifier_type == "rf":
        depth = trial.suggest_int("rf_depth", 2, 32, log=True)
        n_estimators = trial.suggest_int("rf_n", 100, 300)
        clf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators, class_weight="balanced", random_state=42)

    elif classifier_type == "xgb":
        param = {
            "n_estimators": trial.suggest_int("xgb_n", 100, 300),
            "max_depth": trial.suggest_int("xgb_d", 3, 9),
            "learning_rate": trial.suggest_float("xgb_lr", 1e-3, 0.1, log=True),
            "scale_pos_weight": trial.suggest_float("xgb_spw", 1.0, 20.0),
            "random_state": 42
        }
        clf = XGBClassifier(**param)

    elif classifier_type == "adab":
        base_depth = trial.suggest_int("adab_depth", 1, 3)
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=base_depth),
            n_estimators=trial.suggest_int("adab_n", 50, 300),
            learning_rate=trial.suggest_float("adab_lr", 0.01, 1.0, log=True),
            random_state=42
        )

    scaler = StandardScaler()
    pipeline = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler),
        ('sampler', sampler),
        ('classifier', clf)
    ])

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            fold_score = f1_score(y_val, y_pred, average='macro')
        except Exception as e:
            print(f"Fold failed: {e}")
            return 0.0

        fold_scores.append(fold_score)

        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)


study = optuna.create_study(
    study_name="stroke_prediction_optimization",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=random_seed, n_startup_trials=10),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
)

study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)

print("\n--- Optimization Finished ---")
print(f"Best F1-macro Score: {study.best_value:.4f}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

df_trials = study.trials_dataframe()
print("\nBest sampler:", df_trials.groupby("params_sampler_type")["value"].mean().idxmax())
print("Best classifier:", df_trials.groupby("params_classifier_type")["value"].mean().idxmax())