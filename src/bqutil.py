"""Thin BigQuery helpers. All heavy aggregation happens server-side; only
admission-level tables ever land locally."""
from __future__ import annotations

import os
import pathlib

import pandas as pd
from google.cloud import bigquery

PROJECT = os.environ.get("LABPID_PROJECT", "labmae")
DATASET = os.environ.get("LABPID_DATASET", "labpid")
ROOT = pathlib.Path(__file__).resolve().parent.parent

_client: bigquery.Client | None = None


def client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT)
    return _client


def q(sql: str, **params) -> pd.DataFrame:
    """Run a query and return a DataFrame. `{ds}` expands to the working dataset."""
    sql = sql.format(ds=f"{PROJECT}.{DATASET}", **params)
    return client().query(sql).to_dataframe()


def run(sql: str, **params) -> None:
    """Execute DDL/DML with no result set."""
    sql = sql.format(ds=f"{PROJECT}.{DATASET}", **params)
    client().query(sql).result()


def cache(name: str, sql: str, refresh: bool = False, **params) -> pd.DataFrame:
    """Query, cached to out/<name>.parquet."""
    path = ROOT / "out" / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = q(sql, **params)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
