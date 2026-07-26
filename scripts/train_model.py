"""
Phase 3 - Section 1 & 2: Model Development, Training and Evaluation
====================================================================
This script is the core "Training Pipeline" modeling step. It:

  1. Loads the train/val/test splits produced by split_data.py
  2. For BOTH tasks supported by this dataset:
       - classification -> target: rating_class (0/1)
       - regression      -> target: rating        (1-5)
     it trains several candidate models with cross-validation +
     hyperparameter search on the training set.
  3. Selects the best model per task based on mean CV score.
  4. Refits the winning model on train+val, evaluates once on the
     held-out test set with task-appropriate metrics.
  5. Saves the trained model (joblib), the fitted feature list, and a
     full model-comparison / evaluation report to outputs/.

Note on data leakage: `price_per_rating` (= price_incl_tax / rating) was
engineered in Phase 2 but is dropped before modeling (see split_data.py)
because it directly encodes the target.

Note on library choice: XGBoost is used when available. In sandboxed /
offline environments without internet access it is not installable, so
this script automatically falls back to scikit-learn's
GradientBoostingClassifier/Regressor, which plays the same role
(boosted trees) for the purpose of model comparison. On a machine with
internet access, `pip install xgboost` will make the script use it
automatically instead.
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                               RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, precision_score, r2_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.path.insert(0, str(Path(__file__).parent))
from split_data import get_feature_columns

BASE_DIR = Path(__file__).parent.parent
SPLIT_DIR = BASE_DIR / "outputs" / "splits"
MODEL_DIR = BASE_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def load_splits():
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    return train, val, test


# ----------------------------------------------------------------------
# Candidate models + small hyperparameter grids per task
# ----------------------------------------------------------------------
def _scaled(estimator, step_name):
    """Wrap a scale-sensitive estimator with a StandardScaler in a Pipeline.
    Feature magnitudes in this dataset vary hugely (e.g. desc_length up to
    ~8600 vs. TF-IDF values < 1), which makes distance/gradient-based models
    (SVM, KNN, LogisticRegression, Ridge) converge slowly or perform poorly
    without scaling. Tree-based models are scale-invariant, so they are left
    unscaled below.
    """
    return Pipeline([("scaler", StandardScaler()), (step_name, estimator)])


def classification_candidates():
    models = {
        "LogisticRegression": (
            _scaled(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), "clf"),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "KNN": (
            _scaled(KNeighborsClassifier(), "clf"),
            {"clf__n_neighbors": [5, 11, 21]},
        ),
        "SVM": (
            _scaled(SVC(probability=True, random_state=RANDOM_STATE), "clf"),
            {"clf__C": [1.0, 10.0], "clf__kernel": ["rbf", "linear"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "max_depth": [None, 8, 16]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
        )
    else:
        models["GradientBoosting"] = (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        )
    return models


def regression_candidates():
    models = {
        "LinearRegression": (_scaled(Ridge(alpha=0.0, random_state=RANDOM_STATE), "reg"), {}),
        "Ridge": (
            _scaled(Ridge(random_state=RANDOM_STATE), "reg"),
            {"reg__alpha": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "max_depth": [None, 8, 16]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
        )
    else:
        models["GradientBoosting"] = (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"n_estimators": [200, 400], "max_depth": [2, 3], "learning_rate": [0.05, 0.1]},
        )
    return models


# ----------------------------------------------------------------------
# Model selection: CV + grid search per candidate, report all results
# ----------------------------------------------------------------------
def run_model_selection(X_train, y_train, task: str):
    print("\n" + "=" * 55)
    print(f"  Model Selection - task: {task}")
    print("=" * 55)

    if task == "classification":
        candidates = classification_candidates()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = "f1"
    else:
        candidates = regression_candidates()
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = "neg_root_mean_squared_error"

    results = {}
    fitted = {}
    for name, (model, grid) in candidates.items():
        if grid:
            search = GridSearchCV(model, grid, cv=cv, scoring=scoring, n_jobs=-1)
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            best_score = search.best_score_
            best_params = search.best_params_
        else:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            best_est = model.fit(X_train, y_train)
            best_score = scores.mean()
            best_params = {}

        display_score = best_score if task == "classification" else -best_score
        metric_name = "F1 (CV mean)" if task == "classification" else "RMSE (CV mean)"
        print(f"  {name:18s} | {metric_name}: {display_score:.4f} | best params: {best_params}")

        results[name] = {"cv_score": best_score, "best_params": best_params}
        fitted[name] = best_est

    # Pick best: highest score (GridSearchCV scoring is already "higher is better",
    # incl. neg_root_mean_squared_error, so max() works for both tasks)
    best_name = max(results, key=lambda n: results[n]["cv_score"])
    print(f"\n  >>> Best model for {task}: {best_name}")
    return best_name, fitted[best_name], results


# ----------------------------------------------------------------------
# Final evaluation on the untouched test set
# ----------------------------------------------------------------------
def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    return {k: round(float(v), 4) for k, v in metrics.items()}


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, y_pred),
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}


def main():
    train, val, test = load_splits()
    feature_cols = get_feature_columns(train)

    # Train on train+val for the final fit (common practice once model/
    # hyperparameters are chosen via CV on train), evaluate once on test.
    train_full = pd.concat([train, val], ignore_index=True)

    X_train = train[feature_cols]
    X_train_full = train_full[feature_cols]
    X_test = test[feature_cols]

    report = {"feature_columns": feature_cols, "n_features": len(feature_cols)}

    # ---------------- Classification: rating_class ----------------
    y_train_cls = train["rating_class"]
    y_train_full_cls = train_full["rating_class"]
    y_test_cls = test["rating_class"]

    best_name_cls, _, cv_results_cls = run_model_selection(X_train, y_train_cls, "classification")
    final_model_cls = dict(classification_candidates())[best_name_cls][0]
    # refit with the tuned params found during selection
    tuned_params_cls = cv_results_cls[best_name_cls]["best_params"]
    final_model_cls.set_params(**tuned_params_cls)
    final_model_cls.fit(X_train_full, y_train_full_cls)
    test_metrics_cls = evaluate_classification(final_model_cls, X_test, y_test_cls)

    joblib.dump(final_model_cls, MODEL_DIR / "best_classifier.pkl")
    print(f"\n✅ Saved best classifier ({best_name_cls}) -> outputs/models/best_classifier.pkl")
    print(f"   Test metrics: {test_metrics_cls}")

    # ---------------- Regression: rating ----------------
    y_train_reg = train["rating"]
    y_train_full_reg = train_full["rating"]
    y_test_reg = test["rating"]

    best_name_reg, _, cv_results_reg = run_model_selection(X_train, y_train_reg, "regression")
    final_model_reg = dict(regression_candidates())[best_name_reg][0]
    tuned_params_reg = cv_results_reg[best_name_reg]["best_params"]
    final_model_reg.set_params(**tuned_params_reg)
    final_model_reg.fit(X_train_full, y_train_full_reg)
    test_metrics_reg = evaluate_regression(final_model_reg, X_test, y_test_reg)

    joblib.dump(final_model_reg, MODEL_DIR / "best_regressor.pkl")
    print(f"\n✅ Saved best regressor ({best_name_reg}) -> outputs/models/best_regressor.pkl")
    print(f"   Test metrics: {test_metrics_reg}")

    # ---------------- Save full report ----------------
    report["classification"] = {
        "target": "rating_class",
        "cv_comparison": {k: {"cv_score": round(float(v["cv_score"]), 4), "best_params": v["best_params"]}
                          for k, v in cv_results_cls.items()},
        "best_model": best_name_cls,
        "test_metrics": test_metrics_cls,
    }
    report["regression"] = {
        "target": "rating",
        "cv_comparison": {k: {"cv_score": round(float(v["cv_score"]), 4), "best_params": v["best_params"]}
                          for k, v in cv_results_reg.items()},
        "best_model": best_name_reg,
        "test_metrics": test_metrics_reg,
    }
    # Overall final model recommendation for the production pipeline
    report["final_model_choice"] = "classification (rating_class) with " + best_name_cls

    with open(BASE_DIR / "outputs" / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n✅ Full evaluation report -> outputs/evaluation_report.json")


if __name__ == "__main__":
    main()
