from pathlib import Path
import json
import mlflow
import mlflow.sklearn
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

def main():
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("Real Estate Investment Advisor")
    cls = json.loads((MODEL_DIR / "classification_metrics.json").read_text(encoding="utf-8"))
    reg = json.loads((MODEL_DIR / "regression_metrics.json").read_text(encoding="utf-8"))
    with mlflow.start_run(run_name="baseline_random_forest_models"):
        for key in ["accuracy", "f1", "roc_auc"]:
            if cls.get(key) is not None:
                mlflow.log_metric(f"classification_{key}", cls[key])
        for key in ["mae_lakhs", "rmse_lakhs", "r2"]:
            if reg.get(key) is not None:
                mlflow.log_metric(f"regression_{key}", reg[key])
        mlflow.log_param("classification_target", "Good_Investment_Proxy")
        mlflow.log_param("regression_target", "current Price_in_Lakhs")
        mlflow.log_param("five_year_forecast", "scenario based; no observed future price label")
        mlflow.log_artifact(str(MODEL_DIR / "model_metadata.json"))
    print("MLflow run logged. Run: mlflow ui --backend-store-uri", (ROOT / "mlruns"))

if __name__ == "__main__":
    main()
