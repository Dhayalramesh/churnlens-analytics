"""Create any missing artifacts (raw data, database, model, predictions) so the app runs from a plain git checkout."""
import sqlite3
import urllib.request
from pathlib import Path

from .config import DB_PATH, EXPORT_PATH, METRICS_PATH, MODEL_PATH, RAW_CSV
from .modeling import ChurnModeler, export_for_bi
from .pipeline import ETLPipeline

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def _has_table(db: Path, name: str) -> bool:
    if not Path(db).exists():
        return False
    with sqlite3.connect(db) as con:
        return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def ensure_artifacts(raw: Path = RAW_CSV, db: Path = DB_PATH, model: Path = MODEL_PATH,
                     metrics: Path = METRICS_PATH, export: Path = EXPORT_PATH) -> list[str]:
    """Do only the steps whose output is missing. Returns what was done (empty list = everything already existed)."""
    done: list[str] = []
    raw, db = Path(raw), Path(db)
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:  # noqa: S310 (fixed https URL)
            raw.write_bytes(response.read())
        done.append("downloaded the raw dataset")
    if not _has_table(db, "customers"):
        ETLPipeline(raw, db).run()
        done.append("ran the ETL pipeline")
    if not (Path(metrics).exists() and _has_table(db, "predictions")):
        ChurnModeler(db, model, metrics).train()
        done.append("trained and evaluated the models")
    if not Path(export).exists():
        export_for_bi(db, export)
        done.append("wrote the BI export")
    return done
