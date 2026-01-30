from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, *, out_csv: Path, out_parquet: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)

