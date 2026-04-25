import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import optuna
import yaml

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_features(path="data/processed/features.csv"):
    print("Loading features...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    return df


def split_data(df, target_col="default", test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits data into train, validation, and test sets.
    Test set is held out and never used during training or tuning.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function — defines the hyperparameter search space.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "random_state": 42,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, preds)


def tune_model(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Runs Optuna hyperparameter search.
    """
    print(f"Tuning XGBoost with Optuna ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    print(f"Best AUC-PR: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """
    Trains final XGBoost model with best hyperparameters.
    """
    print("Training final model...")
    params = {
        **best_params,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "random_state": 42,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 50,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


def evaluate_model(model, X, y, split_name="Test"):
    """
    Evaluates model and prints AUC-ROC and AUC-PR.
    """
    preds = model.predict_proba(X)[:, 1]
    auc_roc = roc_auc_score(y, preds)
    auc_pr = average_precision_score(y, preds)
    print(f"{split_name} — AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")
    return auc_roc, auc_pr


if __name__ == "__main__":
    df = load_features()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    best_params = tune_model(X_train, y_train, X_val, y_val, n_trials=50)

    model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    print("\nEvaluation:")
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model.pkl")
    print("\nModel saved to models/xgboost_model.pkl")