"""SQL analytics layer: a read-only repository plus named queries (joins, CTEs, window functions)."""
import sqlite3
from pathlib import Path

import pandas as pd

from .config import DB_PATH

QUERIES: dict[str, str] = {
    "kpis": """
        SELECT COUNT(*)                                        AS customers,
               ROUND(100.0 * AVG(churn_flag), 2)               AS churn_rate_pct,
               ROUND(AVG(MonthlyCharges), 2)                   AS avg_monthly_charge,
               ROUND(SUM(CASE WHEN churn_flag = 1 THEN MonthlyCharges END), 2) AS monthly_revenue_lost,
               ROUND(SUM(MonthlyCharges), 2)                   AS monthly_revenue_total
        FROM customers
    """,
    "churn_by_contract": """
        SELECT Contract,
               COUNT(*)                          AS customers,
               SUM(churn_flag)                   AS churned,
               ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC
    """,
    "churn_by_tenure": """
        SELECT tenure_bucket,
               COUNT(*)                          AS customers,
               ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
        FROM customers
        GROUP BY tenure_bucket
        ORDER BY MIN(tenure)
    """,
    "riskiest_segments": """
        WITH segments AS (
            SELECT Contract, InternetService, PaymentMethod,
                   COUNT(*)                          AS customers,
                   ROUND(100.0 * AVG(churn_flag), 2) AS churn_rate_pct
            FROM customers
            GROUP BY Contract, InternetService, PaymentMethod
            HAVING COUNT(*) >= 30
        )
        SELECT *
        FROM (
            SELECT segments.*,
                   RANK() OVER (PARTITION BY Contract ORDER BY churn_rate_pct DESC) AS rank_in_contract
            FROM segments
        )
        WHERE rank_in_contract <= 3
        ORDER BY Contract, rank_in_contract
    """,
    "above_contract_average_price": """
        WITH priced AS (
            SELECT customerID, Contract, MonthlyCharges, churn_flag,
                   AVG(MonthlyCharges) OVER (PARTITION BY Contract) AS contract_avg
            FROM customers
        )
        SELECT Contract,
               COUNT(*)                                           AS customers_above_average,
               ROUND(100.0 * AVG(churn_flag), 2)                  AS churn_rate_pct
        FROM priced
        WHERE MonthlyCharges > contract_avg
        GROUP BY Contract
        ORDER BY churn_rate_pct DESC
    """,
    "revenue_at_risk_by_contract": """
        SELECT c.Contract,
               COUNT(*)                                             AS customers,
               ROUND(SUM(c.MonthlyCharges * p.churn_probability), 2) AS expected_monthly_revenue_at_risk
        FROM customers AS c
        JOIN predictions AS p ON p.customerID = c.customerID
        WHERE c.churn_flag = 0
        GROUP BY c.Contract
        ORDER BY expected_monthly_revenue_at_risk DESC
    """,
}


class SQLRepository:
    """Runs SQL against the project database. Opens it read-only so a query can never change data."""

    def __init__(self, db_path: Path = DB_PATH, read_only: bool = True):
        self.db_path = Path(db_path)
        self.read_only = read_only

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            con.execute("PRAGMA query_only = ON")
            return con
        return sqlite3.connect(self.db_path)

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        con = self._connect()
        try:
            return pd.read_sql_query(sql, con, params=params)
        finally:
            con.close()

    def named(self, name: str) -> pd.DataFrame:
        return self.query(QUERIES[name])
