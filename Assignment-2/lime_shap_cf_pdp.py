
import shap
import dice_ml
from dice_ml import Dice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from xgboost import XGBRegressor

# lime
from interpret import show
from interpret.blackbox import LimeTabular

# pdp
from sklearn.inspection import PartialDependenceDisplay


'''
The objective of this script is to create three black-box ML models for
explaining them locally using LIME in an interactive dashboard. 
The black-box models to explain are a neural network, an 
XGB and a random forest. 

Data will be splitted into training and testing, without any subset for 
validation in order to stay simple, avoiding model optimization, as the
task is to understand why are the models taking decissions more than 
improving the decissions themself.

Only the first 250 predictions will be shown by the lime dashboard,
but this parameter can be changed with the global variable SHOW_PREDS.
Additionally models performance will also be shown.
'''




RANDOM_SEED = 42
DATA_PATH = "./Assignment-2/LengthOfStay.csv"
PROCESSED_PATH = "./Assignment-2/processed_data.csv"
SHOW_PREDS = 250 

PREPROCESS_DATA = False
LIME = False
SHAP = False
COUNTERFACTUAL = False
PDP = True

CF_INSTANCES   = 3   # how many test rows to explain
CF_PER_INSTANCE = 3  # how many counterfactuals to generate each time


if PREPROCESS_DATA:
    data = pd.read_csv(DATA_PATH) # .iloc[0:5000,:]# for faster rendering

    # Data cleaning
    order = ["0", "1", "2", "3", "4", "5+"]
    data['rcount'] = pd.Categorical(data['rcount'], categories=order, ordered=True)
    data['rcount'] = data['rcount'].cat.codes  # 0,1,2,3,4,5 integers

    categorical_cols = ['gender', 'facid']

    # feature engineering
    non_feature_cols = {'eid', 'vdate', 'discharged', 'lengthofstay'} | set(categorical_cols)
    binary_cols = [
        col for col in data.columns
        if col not in non_feature_cols and data[col].dropna().isin([0, 1]).all()
    ]
    data['bin_col_sum'] = data[binary_cols].sum(axis=1)

    df = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    df = df.drop(columns=['eid', 'vdate', 'discharged'])

    df.to_csv(PROCESSED_PATH)

else:
    df = pd.read_csv(PROCESSED_PATH) # .iloc[0:5000,:]# for faster rendering


# Data splitting
X = df.drop(columns='lengthofstay')
y = df['lengthofstay']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
feature_names = X.columns.tolist()


# Models creation

neural_network = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(15,15,15,15), # 3 hidden layers with 10 hidden units in each
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=RANDOM_SEED,
        early_stopping=True,
        n_iter_no_change=50
    )
)

random_forest = make_pipeline(
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=150,
        random_state=RANDOM_SEED,
    )
)

xgb = make_pipeline(
    StandardScaler(),
    XGBRegressor(
        n_estimators=150,
        random_state=RANDOM_SEED,
        eval_metric='logloss'
    )
)

models = {'nn': neural_network, 'rf': random_forest, 'xgb' :xgb}

# Model training
for model in models:
    models[model].fit(X_train, y_train)


# Model Results
print("PERFORMANCE METRICS:\n")
for model in models:
    preds = models[model].predict(X_test)

    print(f"{model}:\n\tRMSE: {root_mean_squared_error(y_test, preds)}\n\tR2: {r2_score(y_test, preds)}\n")



# LIME
if LIME:
    for model in models:
        print(f"Showing the model: {model}")

        lime = LimeTabular(
            model = models[model],
            data=X_train,
            random_state=RANDOM_SEED,
            mode = 'regression'
        )

        lime_local = lime.explain_local(
            X_test[:SHOW_PREDS],
            y_test[:SHOW_PREDS],
            name='lime'
        )

        show(lime_local)
        input("")  # press enter to see the next model


# SHAP
if SHAP:
    X_test_explain = X_test.iloc[:SHOW_PREDS]

    for name, model in models.items():
        print(f"""\n── SHAP for "{name}" ──""")

        scaler: StandardScaler = model.named_steps["standardscaler"]
        estimator = model[-1]  # final step after scaler

        # Scale the slices we need
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train), columns=feature_names
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_explain), columns=feature_names
        )

        if name in ("rf", "xgb"):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer(X_test_scaled)

        else:
            background = shap.sample(X_train_scaled, 100, random_state=RANDOM_SEED)
            explainer = shap.KernelExplainer(
                estimator.predict, background, link="identity"
            )
            
            raw = explainer.shap_values(X_test_scaled, nsamples=200)
            shap_values = shap.Explanation(
                values=raw,
                base_values=np.full(len(raw), explainer.expected_value),
                data=X_test_scaled.values,
                feature_names=feature_names
            )

        # ── Plot 1: global feature importance (beeswarm) ────────────────────────
        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        plt.title(f"SHAP Beeswarm — {name}")
        plt.tight_layout()
        plt.savefig(f"shap_beeswarm_{name}.png", dpi=150, bbox_inches="tight")
        plt.show()

        # ── Plot 2: bar summary (mean |SHAP|) ────────────────────────────────────
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=15, show=False)
        plt.title(f"SHAP Feature Importance — {name}")
        plt.tight_layout()
        plt.savefig(f"shap_bar_{name}.png", dpi=150, bbox_inches="tight")
        plt.show()

        # ── Plot 3: waterfall for the first test instance ────────────────────────
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title(f"SHAP Waterfall (instance 0) — {name}")
        plt.tight_layout()
        plt.savefig(f"shap_waterfall_{name}.png", dpi=150, bbox_inches="tight")
        plt.show()

        input(f"  [Enter] to continue to next model…\n")

# Counterfactual Explanations (DiCE)
if COUNTERFACTUAL:
    dummy_and_ordinal_cols = [c for c in feature_names if
                               c.startswith("gender_") or
                               c.startswith("facid_") or
                               c == "rcount"]
    continuous_cols = [c for c in feature_names if c not in dummy_and_ordinal_cols]

    dice_data = dice_ml.Data(
        dataframe=pd.concat([X_train, y_train], axis=1),
        continuous_features=continuous_cols,
        outcome_name="lengthofstay"
    )

    cf_input = X_test.iloc[:CF_INSTANCES].reset_index(drop=True)

    for name, model in models.items():
        print(f"""\n── Counterfactuals for "{name}" ──""")

        dice_model = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
        explainer_cf = Dice(dice_data, dice_model, method="random")

        current_preds = model.predict(cf_input)
        print("Current predictions:", np.round(current_preds, 2))
        
        mean_pred = float(np.mean(current_preds))
        desired_range = [max(1, mean_pred * 0.6), mean_pred * 0.85]
        print(f"Desired range: {[round(x,2) for x in desired_range]}")

        cf = explainer_cf.generate_counterfactuals(
            cf_input,
            total_CFs=CF_PER_INSTANCE,
            desired_range=desired_range,
            features_to_vary=continuous_cols,  # only vary continuous features
            random_seed=RANDOM_SEED,
        )

        # ── Text summary ────────────────────────────────────────────────────
        cf.visualize_as_dataframe(show_only_changes=True)

        # ── Save each instance's counterfactuals as CSV ──────────────────────
        for i, cf_example in enumerate(cf.cf_examples_list):
            df_cf = cf_example.final_cfs_df
            if df_cf is not None and not df_cf.empty:
                original_row = cf_input.iloc[[i]].copy()
                original_row["lengthofstay"] = round(current_preds[i], 3)
                original_row["_type"] = "original"
                df_cf["_type"] = "counterfactual"
                combined = pd.concat([original_row, df_cf], ignore_index=True)
                path = f"cf_{name}_instance{i}.csv"
                combined.to_csv(path, index=False)
                print(f"  Saved: {path}")

        input(f"  [Enter] to continue to next model…\n")


if PDP:
    TOP_FEATURES = 5  

    for name, model in models.items():
        print(f"""\n── PDP for "{name}" ──""")

        scaler: StandardScaler = model.named_steps["standardscaler"]
        estimator = model[-1]

        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=feature_names
        )

        features_to_plot = feature_names[:TOP_FEATURES]

        fig, ax = plt.subplots(figsize=(10, 6))

        PartialDependenceDisplay.from_estimator(
            estimator,
            X_train_scaled,
            features=features_to_plot,
            kind="both",   # avg and individuals
            ax=ax
        )

        plt.suptitle(f"Partial Dependence — {name}")
        plt.tight_layout()
        plt.savefig(f"pdp_{name}.png", dpi=150, bbox_inches="tight")
        plt.show()

        input(f"  [Enter] to continue to next model…\n")