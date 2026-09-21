"""Object-oriented ETL: extract the raw CSV, validate, clean and enrich it, and load it into SQLite."""
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .algorithms import bucketize
from .config import DB_PATH, RAW_CSV

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]
SERVICE_COLUMNS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
TENURE_BOUNDS = [12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months", "73+ months"]


@dataclass
class ValidationReport:
    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)


class DataValidator:
    """Data-quality gates. The pipeline stops if a check fails, instead of loading bad data."""

    def validate_raw(self, df: pd.DataFrame) -> ValidationReport:
        r = ValidationReport()
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        r.add("required columns present", not missing, f"missing: {missing}" if missing else f"{len(REQUIRED_COLUMNS)} columns")
        r.add("has rows", len(df) > 0, f"{len(df)} rows")
        if missing:
            return r
        dup = int(df["customerID"].duplicated().sum())
        r.add("customerID is unique", dup == 0, f"{dup} duplicates")
        bad_churn = sorted(set(df["Churn"].dropna().astype(str).str.strip()) - {"Yes", "No"})
        r.add("Churn is Yes/No", not bad_churn, f"unexpected values: {bad_churn}" if bad_churn else "ok")
        r.add("tenure is non-negative", bool((df["tenure"] >= 0).all()), f"min={df['tenure'].min()}")
        r.add("MonthlyCharges is non-negative", bool((df["MonthlyCharges"] >= 0).all()), f"min={df['MonthlyCharges'].min()}")
        return r

    def validate_clean(self, df: pd.DataFrame) -> ValidationReport:
        r = ValidationReport()
        key_nulls = int(df[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "churn_flag"]].isna().sum().sum())
        r.add("no nulls in key columns after cleaning", key_nulls == 0, f"{key_nulls} nulls")
        r.add("TotalCharges is numeric and non-negative", bool((df["TotalCharges"] >= 0).all()), f"min={df['TotalCharges'].min()}")
        r.add("churn_flag is 0/1", set(df["churn_flag"].unique()) <= {0, 1}, f"values={sorted(int(v) for v in df['churn_flag'].unique())}")
        r.add("tenure_bucket assigned to every row", bool(df["tenure_bucket"].notna().all()), "ok")
        return r


@dataclass
class ETLResult:
    rows_in: int
    rows_out: int
    raw_report: ValidationReport
    clean_report: ValidationReport
    seconds: dict
    db_path: Path


class ETLPipeline:
    """extract -> validate -> transform -> validate -> load. Mirrors the stages of a managed ETL job."""

    def __init__(self, raw_path: Path = RAW_CSV, db_path: Path = DB_PATH):
        self.raw_path = Path(raw_path)
        self.db_path = Path(db_path)
        self.validator = DataValidator()

    def extract(self) -> pd.DataFrame:
        return pd.read_csv(self.raw_path)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:                       # trim stray whitespace in text columns
            if pd.api.types.is_string_dtype(out[col]):
                out[col] = out[col].str.strip()
        # TotalCharges arrives as text; blanks belong to customers with tenure 0 (not billed yet)
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        new_customers = out["TotalCharges"].isna() & (out["tenure"] == 0)
        out.loc[new_customers, "TotalCharges"] = 0.0
        still_missing = out["TotalCharges"].isna()     # any other gap: estimate from tenure x monthly price
        out.loc[still_missing, "TotalCharges"] = out.loc[still_missing, "tenure"] * out.loc[still_missing, "MonthlyCharges"]
        out["churn_flag"] = (out["Churn"] == "Yes").astype(int)
        out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
        out["num_services"] = out[SERVICE_COLUMNS].isin(["Yes"]).sum(axis=1) + (out["InternetService"] != "No").astype(int)
        out["tenure_bucket"] = out["tenure"].map(lambda t: bucketize(t, TENURE_BOUNDS, TENURE_LABELS))
        return out

    def load(self, df: pd.DataFrame) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            df.to_sql("customers", con, if_exists="replace", index=False)
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_customers_id ON customers(customerID)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_customers_contract ON customers(Contract)")

    def run(self) -> ETLResult:
        timings: dict[str, float] = {}

        def timed(name, fn, *args):
            start = time.perf_counter()
            value = fn(*args)
            timings[name] = round(time.perf_counter() - start, 4)
            return value

        raw = timed("extract", self.extract)
        raw_report = self.validator.validate_raw(raw)
        if not raw_report.passed:
            raise ValueError(f"raw data failed validation:\n{raw_report.to_frame().to_string(index=False)}")
        clean = timed("transform", self.transform, raw)
        clean_report = self.validator.validate_clean(clean)
        if not clean_report.passed:
            raise ValueError(f"clean data failed validation:\n{clean_report.to_frame().to_string(index=False)}")
        timed("load", self.load, clean)
        return ETLResult(len(raw), len(clean), raw_report, clean_report, timings, self.db_path)
