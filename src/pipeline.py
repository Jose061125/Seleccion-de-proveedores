from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text


def split_features_target(df: pd.DataFrame, target_col: str, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and target vector y."""
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


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
    """Train a decision tree pipeline with preprocessing and return useful artifacts."""
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
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
    }


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


def feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Return sorted feature importances from the trained decision tree."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({"feature": names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def decision_rules(pipeline: Pipeline) -> str:
    """Export decision rules as plain text for interpretability."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out().tolist()
    return export_text(model, feature_names=names)
