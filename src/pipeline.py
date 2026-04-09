from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

MODEL_LABELS = {
    "decision_tree": "Arbol de decision",
    "random_forest": "Random Forest",
    "logistic_regression": "Regresion logistica",
}


def split_features_target(df: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and target vector y."""
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical features."""
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_estimator(
    model_name: str,
    *,
    random_state: int = 42,
    max_depth: int | None = None,
    criterion: str = "gini",
    min_samples_split: int = 2,
) -> Any:
    """Create estimator based on selected model name."""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=random_state)

    raise ValueError(f"Modelo no soportado: {model_name}")


def build_pipeline(
    X: pd.DataFrame,
    *,
    model_name: str = "decision_tree",
    random_state: int = 42,
    max_depth: int | None = None,
    criterion: str = "gini",
    min_samples_split: int = 2,
) -> Pipeline:
    """Build a full preprocessing + estimator pipeline."""
    preprocessor = build_preprocessor(X)
    model = build_estimator(
        model_name,
        random_state=random_state,
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str = "decision_tree",
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int | None = None,
    criterion: str = "gini",
    min_samples_split: int = 2,
) -> dict[str, Any]:
    """Train a selected model pipeline and return useful artifacts."""
    pipeline = build_pipeline(
        X,
        model_name=model_name,
        random_state=random_state,
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
    )

    stratify_y = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    pipeline.fit(X_train, y_train)

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model_name": model_name,
        "model_label": MODEL_LABELS[model_name],
    }


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int | None = None,
    criterion: str = "gini",
    min_samples_split: int = 2,
) -> dict[str, Any]:
    """Backward-compatible wrapper for training a decision tree."""
    return train_model(
        X,
        y,
        model_name="decision_tree",
        test_size=test_size,
        random_state=random_state,
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
    )


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate the trained model and return scalar metrics and report."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T

    confusion_matrix = pd.crosstab(
        pd.Series(y_test, name="Real"),
        pd.Series(y_pred, name="Prediccion"),
        margins=True,
    )

    return {
        "metrics": metrics,
        "report_df": report_df,
        "confusion_matrix": confusion_matrix,
    }


def cross_validation_summary(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str = "decision_tree",
    cv_folds: int = 5,
    random_state: int = 42,
    max_depth: int | None = None,
    criterion: str = "gini",
    min_samples_split: int = 2,
) -> pd.DataFrame:
    """Run cross validation and summarize mean and std metrics."""
    pipeline = build_pipeline(
        X,
        model_name=model_name,
        random_state=random_state,
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=None)

    rows = []
    for metric_name in scoring:
        scores = results[f"test_{metric_name}"]
        rows.append(
            {
                "metrica": metric_name,
                "media": float(np.mean(scores)),
                "desviacion": float(np.std(scores)),
                "minimo": float(np.min(scores)),
                "maximo": float(np.max(scores)),
            }
        )

    return pd.DataFrame(rows)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int = 5,
    random_state: int = 42,
    max_depth: int | None = None,
    criterion: str = "gini",
    min_samples_split: int = 2,
) -> pd.DataFrame:
    """Compare main candidate models with cross validation."""
    rows = []
    for model_name in MODEL_LABELS:
        summary_df = cross_validation_summary(
            X,
            y,
            model_name=model_name,
            cv_folds=cv_folds,
            random_state=random_state,
            max_depth=max_depth if model_name != "logistic_regression" else None,
            criterion=criterion,
            min_samples_split=min_samples_split,
        )
        summary_by_metric = summary_df.set_index("metrica")["media"].to_dict()
        rows.append(
            {
                "modelo": MODEL_LABELS[model_name],
                "accuracy": summary_by_metric.get("accuracy", 0.0),
                "precision_weighted": summary_by_metric.get("precision_weighted", 0.0),
                "recall_weighted": summary_by_metric.get("recall_weighted", 0.0),
                "f1_weighted": summary_by_metric.get("f1_weighted", 0.0),
            }
        )

    return pd.DataFrame(rows).sort_values("f1_weighted", ascending=False).reset_index(drop=True)


def feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Return sorted feature importances or coefficients from the trained model."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({"feature": names, "importance": importances})
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def decision_rules(pipeline: Pipeline) -> str:
    """Export decision rules as plain text for interpretability."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if not isinstance(model, DecisionTreeClassifier):
        return "Las reglas detalladas solo estan disponibles para el modelo Arbol de decision."

    names = preprocessor.get_feature_names_out().tolist()
    return export_text(model, feature_names=names)


def tree_visualization_figure(pipeline: Pipeline, *, max_depth: int = 3):
    """Create a matplotlib figure for a trained decision tree."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if not isinstance(model, DecisionTreeClassifier):
        return None

    feature_names = preprocessor.get_feature_names_out().tolist()
    class_names = [str(item) for item in model.classes_]
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        max_depth=max_depth,
        ax=ax,
    )
    fig.tight_layout()
    return fig
