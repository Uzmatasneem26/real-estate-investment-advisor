from pathlib import Path
import json
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned_housing_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def make_investment_label(df: pd.DataFrame) -> pd.Series:
    # Proxy label only: actual return/appreciation labels are absent from source data.
    city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
    relative_value = df["Price_per_SqFt"] <= city_median
    good_access = df["Public_Transport_Accessibility"].astype(str).str.lower().isin(["high", "medium"])
    amenity_score = df.get("Amenities_Count", pd.Series(0, index=df.index)) >= 2
    newer = df["Age_of_Property"] <= df["Age_of_Property"].median()
    return (relative_value & good_access & (amenity_score | newer)).astype(int)

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run python scripts/data_cleaning.py first.")
    df = pd.read_csv(DATA)
    df["Good_Investment_Proxy"] = make_investment_label(df)

    # Use current price as the supervised regression target. The app applies a user-selected
    # appreciation scenario to this estimated current value; it is not trained on real future prices.
    drop_cols = [c for c in ["ID", "Good_Investment_Proxy", "Price_in_Lakhs", "Price_per_SqFt"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")
    y_cls = df["Good_Investment_Proxy"]
    y_reg = df["Price_in_Lakhs"]

    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=180, max_depth=18, min_samples_leaf=2,
                                         class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    cls_metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() == 2 else None,
        "target_note": "Proxy label created from relative city price, transport access, amenities, and property age. Not actual investment returns."
    }
    joblib.dump(clf, MODEL_DIR / "investment_model.pkl")
    (MODEL_DIR / "classification_metrics.json").write_text(json.dumps(cls_metrics, indent=2), encoding="utf-8")

    X_train_r, X_test_r, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    reg = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
        ])),
        ("model", RandomForestRegressor(n_estimators=180, max_depth=20, min_samples_leaf=2,
                                        random_state=42, n_jobs=-1))
    ])
    reg.fit(X_train_r, yr_train)
    pred_r = reg.predict(X_test_r)
    reg_metrics = {
        "mae_lakhs": float(mean_absolute_error(yr_test, pred_r)),
        "rmse_lakhs": float(np.sqrt(mean_squared_error(yr_test, pred_r))),
        "r2": float(r2_score(yr_test, pred_r)),
        "target_note": "Model predicts current listed price in lakhs. The five-year forecast is scenario-based because no observed future-price target exists."
    }
    joblib.dump(reg, MODEL_DIR / "price_prediction_model.pkl")
    (MODEL_DIR / "regression_metrics.json").write_text(json.dumps(reg_metrics, indent=2), encoding="utf-8")
    metadata = {
        "feature_columns": X.columns.tolist(),
        "categorical_columns": categorical,
        "numerical_columns": numerical,
        "classification_target": "Good_Investment_Proxy",
        "regression_target": "Price_in_Lakhs",
        "dataset_rows": int(len(df))
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Models saved in", MODEL_DIR)
    print("Classification metrics:", cls_metrics)
    print("Regression metrics:", reg_metrics)

if __name__ == "__main__":
    main()
