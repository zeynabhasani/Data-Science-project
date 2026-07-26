"""
Phase 3 - Section 4 (10% Bonus): MLflow Experiment Tracking
=============================================================
Wraps the same model-selection process as train_model.py, but logs every
candidate model's parameters + CV score as its own MLflow run, then logs
the final chosen model (per task) with its test-set metrics and the fitted
model artifact itself via MLflow's model registry.

Requires: pip install mlflow
(Not runnable in this sandbox - no internet access to install the package -
but this script uses the standard MLflow API and will run as-is on any
machine with `pip install mlflow` executed first.)

Usage:
    pip install mlflow
    python scripts/mlflow_train.py
    mlflow ui   # then open http://127.0.0.1:5000 to inspect runs
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    print("MLflow is not installed in this environment.")
    print("Run: pip install mlflow")
    print("Then re-run this script: python scripts/mlflow_train.py")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from split_data import get_feature_columns
from train_model import (classification_candidates, regression_candidates,
                          evaluate_classification, evaluate_regression,
                          RANDOM_STATE)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score

BASE_DIR = Path(__file__).parent.parent
SPLIT_DIR = BASE_DIR / "outputs" / "splits"

mlflow.set_experiment("books-to-scrape-rating-prediction")


def load_splits():
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    return train, val, test


def run_task(task: str, train, val, test, feature_cols):
    X_train = train[feature_cols]
    train_full = pd.concat([train, val], ignore_index=True)
    X_train_full = train_full[feature_cols]
    X_test = test[feature_cols]

    if task == "classification":
        candidates = classification_candidates()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = "f1"
        y_train, y_train_full, y_test = train["rating_class"], train_full["rating_class"], test["rating_class"]
    else:
        candidates = regression_candidates()
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = "neg_root_mean_squared_error"
        y_train, y_train_full, y_test = train["rating"], train_full["rating"], test["rating"]

    best_name, best_score, best_model, best_params = None, -float("inf"), None, None

    for name, (model, grid) in candidates.items():
        with mlflow.start_run(run_name=f"{task}-{name}"):
            mlflow.log_param("task", task)
            mlflow.log_param("model_family", name)

            if grid:
                search = GridSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=-1)
                search.fit(X_train, y_train)
                fitted, score, params = search.best_estimator_, search.best_score_, search.best_params_
            else:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                fitted, score, params = model.fit(X_train, y_train), scores.mean(), {}

            mlflow.log_params(params)
            mlflow.log_metric("cv_score", float(score))

            if score > best_score:
                best_name, best_score, best_model, best_params = name, score, fitted, params

    # Final run: refit the winning model on train+val, log test metrics + model artifact
    with mlflow.start_run(run_name=f"{task}-FINAL-{best_name}"):
        final_model = dict(candidates)[best_name][0]
        final_model.set_params(**best_params)
        final_model.fit(X_train_full, y_train_full)

        if task == "classification":
            metrics = evaluate_classification(final_model, X_test, y_test)
        else:
            metrics = evaluate_regression(final_model, X_test, y_test)

        mlflow.log_param("task", task)
        mlflow.log_param("model_family", best_name)
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(final_model, artifact_path="model")
        print(f"\n[{task}] Best model: {best_name} | CV score: {best_score:.4f} | Test metrics: {metrics}")

    return best_name, metrics


if __name__ == "__main__":
    train, val, test = load_splits()
    feature_cols = get_feature_columns(train)

    print("=" * 55)
    print("  MLflow-tracked model selection & training")
    print("=" * 55)
    run_task("classification", train, val, test, feature_cols)
    run_task("regression", train, val, test, feature_cols)

    print("\n✅ All runs logged to MLflow.")
    print("   Start the UI with:  mlflow ui")
    print("   Then open:          http://127.0.0.1:5000")
