"""Central paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "processed" / "churnlens.db"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
EXPORT_PATH = ROOT / "exports" / "customers_for_powerbi.csv"
SEED = 42
