
import optuna
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

def objective(trial):
    # imputation
    imputer_type = trial.suggest_categorical("imputer_type", ["simple", "knn", "iterative"])
    if imputer_type == "simple":
        strat = trial.suggest_categorical("simp_imp_strategy", ["mean", "median", "most_frequent"])
        imputer = SimpleImputer(strategy=strat)
    elif imputer_type == "knn":
        n_neighbors = trial.suggest_int("knn_input_neighs", 3, 20)
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer = IterativeImputer(random_state=random_seed)

    # sampling
    sampler_type = trial.suggest_categorical(
        "sampler_type",
        ["random_over", "random_under", "adasyn", "smote", "smoteenn",
         "smotetomek", "svmsmote", "borderlinesmote", "kmeansmote", "none"]
    
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
    elif sampler_type == "smoteenn":
        sampler = SMOTEENN(random_state=random_seed)
    elif sampler_type == "smotetomek":
        sampler = SMOTETomek(random_state=random_seed)
    elif sampler_type == "svmsmote":
        k_neighs = trial.suggest_int("svmsmote_k_neighs", 3, 20)
        m_neighs = trial.suggest_int("svmsmote_m_neighs", 5, 50)
        sampler = SVMSMOTE(
            k_neighbors=k_neighs,
            m_neighbors=m_neighs,
            random_state=random_seed
        )
    elif sampler_type == "borderlinesmote":
        k_neighs = trial.suggest_int("borderlinesmote_k_neighs", 3, 20)
        m_neighs = trial.suggest_int("borderlinesmote_m_neighs", 5, 50)
        kind = trial.suggest_categorical("borderlinesmote_kind", ["borderline-1", "borderline-2"])
        sampler = BorderlineSMOTE(
            k_neighbors=k_neighs,
            m_neighbors=m_neighs,
            kind=kind,
            random_state=random_seed
        )
    elif sampler_type == "kmeansmote":
        k_neighs = trial.suggest_int("kmeansmote_k_neighs", 3, 20)
        sampler = KMeansSMOTE(
            k_neighbors=k_neighs,
            random_state=random_seed
        )
    else:
        sampler = "passthrough"

    # classifier
    classifier_type = trial.suggest_categorical(
        "classifier_type",
        ["lr", "rf", "xgb", "adab", "bayes", "svc", "knn"]
    )
    if classifier_type == "lr":
        c_param = trial.suggest_float("lr_c", 1e-5, 10, log=True)
        clf = LogisticRegression(C=c_param, max_iter=1000, random_state=42)

    elif classifier_type == "rf":
        depth = trial.suggest_int("rf_depth", 2, 32, log=True)
        n_estimators = trial.suggest_int("rf_n", 50, 500)
        clf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators, random_state=42)

    elif classifier_type == "xgb":
        param = {
            "n_estimators": trial.suggest_int("xgb_n", 100, 500),
            "max_depth": trial.suggest_int("xgb_d", 3, 9),
            "learning_rate": trial.suggest_float("xgb_lr", 1e-3, 0.1, log=True),
            "random_state": 42
        }
        clf = XGBClassifier(**param)

    elif classifier_type == "adab":
        # AdaBoost often works best with shallow decision trees (stumps)
        base_depth = trial.suggest_int("adab_depth", 1, 3)
        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=base_depth),
            n_estimators=trial.suggest_int("adab_n", 50, 500),
            learning_rate=trial.suggest_float("adab_lr", 0.01, 1.0, log=True),
            random_state=42
        )

    elif classifier_type == "bayes":
        # GaussianNB has fewer params, but we can tune the var_smoothing
        smoothing = trial.suggest_float("nb_smooth", 1e-11, 1e-8, log=True)
        clf = GaussianNB(var_smoothing=smoothing)

    elif classifier_type == "svc":
        c_svc = trial.suggest_float("svc_c", 1e-4, 10, log=True)
        kernel = trial.suggest_categorical("svc_kernel", ["linear", "poly", "rbf"])
        clf = SVC(C=c_svc, kernel=kernel, probability=True, random_state=42)

    elif classifier_type == "knn":
        k = trial.suggest_int("knn_k", 3, 15)
        weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        clf = KNeighborsClassifier(n_neighbors=k, weights=weights)

    scaler = StandardScaler()
    pipeline = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler),
        ('sampler', sampler),
        ('classifier', clf)
    ])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
    
    try:
        score = cross_val_score(
            pipeline, 
            df_dummies.drop(columns=['stroke']), 
            df_dummies['stroke'], 
            cv=skf, 
            scoring='f1_macro',
            n_jobs=-1
        ).mean()
    except Exception as e:
        print(f"Trial failed due to: {e}")
        return 0.0
    return score

# 1. Create the study
# We use 'maximize' because we want to maximize the F1-score
study = optuna.create_study(
    study_name="stroke_prediction_optimization",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=random_seed) # Uses Bayesian optimization
)

study.optimize(objective, n_trials=100, show_progress_bar=True)

print("\n--- Optimization Finished ---")
print(f"Best F1-macro Score: {study.best_value:.4f}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")