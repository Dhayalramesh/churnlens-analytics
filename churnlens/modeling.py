"""Machine-learning layer: preprocessing, model comparison, threshold tuning, evaluation and saving."""
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DB_PATH, METRICS_PATH, MODEL_PATH, SEED

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "num_services"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
               "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod", "tenure_bucket"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn_flag"


def score_metrics(y_true, proba, threshold: float) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y_true, proba) -> float:
    """Sweep thresholds and return the one with the highest F1 (run on out-of-fold predictions, never on the test set)."""
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


class ChurnModeler:
    def __init__(self, db_path: Path = DB_PATH, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH, seed: int = SEED):
        self.db_path, self.model_path, self.metrics_path, self.seed = Path(db_path), Path(model_path), Path(metrics_path), seed

    def load_frame(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as con:
            return pd.read_sql_query("SELECT * FROM customers", con)

    @staticmethod
    def _preprocessor(scale: bool) -> ColumnTransformer:
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])

    def candidates(self) -> dict[str, Pipeline]:
        return {
            "Logistic Regression": Pipeline([("prep", self._preprocessor(True)),
                                             ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            "Random Forest": Pipeline([("prep", self._preprocessor(False)),
                                       ("model", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                                        class_weight="balanced", n_jobs=-1, random_state=self.seed))]),
            "Hist Gradient Boosting": Pipeline([("prep", self._preprocessor(False)),
                                                ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                                                                         class_weight="balanced", random_state=self.seed))]),
        }

    def train(self, save: bool = True) -> dict:
        df = self.load_frame()
        X, y = df[FEATURES], df[TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        # 1) compare models with cross-validation on the training split only
        comparison = []
        for name, pipe in self.candidates().items():
            aucs = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
            comparison.append({"model": name, "cv_roc_auc_mean": round(float(aucs.mean()), 4), "cv_roc_auc_std": round(float(aucs.std()), 4)})
        best_name = max(comparison, key=lambda r: r["cv_roc_auc_mean"])["model"]
        best = self.candidates()[best_name]

        # 2) tune the decision threshold on out-of-fold predictions (no test-set peeking)
        oof_train = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        threshold = best_f1_threshold(y_tr, oof_train)

        # 3) final fit and one honest evaluation on the untouched test split
        best.fit(X_tr, y_tr)
        proba_te = best.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba_te)
        keep = np.linspace(0, len(fpr) - 1, num=min(80, len(fpr))).astype(int)
        imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=self.seed, n_jobs=1)
        importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                      .sort_values("importance", ascending=False).head(10).round(4).to_dict("records"))

        # 4) out-of-fold churn probability for EVERY customer (honest scores for the dashboard)
        oof_all = cross_val_predict(self.candidates()[best_name], X, y, cv=cv, method="predict_proba")[:, 1]
        predictions = pd.DataFrame({"customerID": df["customerID"], "churn_probability": np.round(oof_all, 4),
                                    "predicted_churn": (oof_all >= threshold).astype(int)})

        metrics = {
            "dataset": {"rows": int(len(df)), "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
                        "churn_rate_pct": round(float(y.mean() * 100), 2)},
            "comparison": comparison, "best_model": best_name, "tuned_threshold": round(threshold, 3),
            "test_at_0.5": score_metrics(y_te, proba_te, 0.5),
            "test_at_tuned_threshold": score_metrics(y_te, proba_te, threshold),
            "roc_curve": {"fpr": [round(float(v), 4) for v in fpr[keep]], "tpr": [round(float(v), 4) for v in tpr[keep]]},
            "feature_importance": importance,
        }
        if save:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best, self.model_path)
            self.metrics_path.write_text(json.dumps(metrics, indent=2))
            with sqlite3.connect(self.db_path) as con:
                predictions.to_sql("predictions", con, if_exists="replace", index=False)
                con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_id ON predictions(customerID)")
        return {"metrics": metrics, "predictions": predictions, "model": best}

    def load_model(self) -> Pipeline:
        """Load the saved model; if the file is missing or from another scikit-learn version, refit quickly."""
        try:
            return joblib.load(self.model_path)
        except Exception:
            name = json.loads(self.metrics_path.read_text())["best_model"]
            df = self.load_frame()
            return self.candidates()[name].fit(df[FEATURES], df[TARGET])


def export_for_bi(db_path: Path, out_path: Path) -> Path:
    """Write one flat table (customers + predictions) that Power BI / Tableau / Excel can open directly."""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(
            "SELECT c.*, p.churn_probability, p.predicted_churn FROM customers c LEFT JOIN predictions p USING (customerID)", con)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
