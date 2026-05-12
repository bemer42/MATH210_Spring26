import numpy as np
import pandas as pd

def build_adj_mats(excel_file):
    """
    Expects an Excel file with columns:
      from, to, 2006, 2007, ..., 2024, Global

    Returns:
      {
        "nodes": [...],
        "A_by_year": {2006: A2006, ..., 2024: A2024},
        "A_global": Aglobal
      }
    """
    years = list(range(2006, 2025))  # 2006..2024

    df = pd.read_excel(
        excel_file,
        usecols=["from", "to"] + [str(y) for y in years] + ["Global"],
        sheet_name=0
    )

    # ensure column names are strings (Excel sometimes reads years as ints)
    df.columns = df.columns.map(lambda c: str(c).strip())

    df["from"] = df["from"].astype(str).str.strip()
    df["to"]   = df["to"].astype(str).str.strip()

    nodes = sorted(pd.unique(pd.concat([df["from"], df["to"]], ignore_index=True)))
    n = len(nodes)

    iu = pd.Categorical(df["from"], categories=nodes).codes
    iv = pd.Categorical(df["to"],   categories=nodes).codes

    A_by_year = {}
    for y in years:
        w = df[str(y)].fillna(0).to_numpy(dtype=float)
        A = np.zeros((n, n), dtype=float)
        A[iu, iv] = w
        A[iv, iu] = w
        A_by_year[y] = A

    w_global = df["Global"].fillna(0).to_numpy(dtype=float)
    A_global = np.zeros((n, n), dtype=float)
    A_global[iu, iv] = w_global
    A_global[iv, iu] = w_global

    return {"nodes": nodes, "A_by_year": A_by_year, "A_global": A_global}