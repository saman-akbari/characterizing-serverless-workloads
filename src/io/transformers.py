import logging
from typing import Callable, List

import dask.dataframe as dd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def _globus24_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.rename(columns={"argument_size": "size"})
    df = df.dropna(subset=["size"])
    return df


def _huawei25_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    df_val, df_req = dfs[0], dfs[1]

    def flatten(pdf_val, pdf_req):
        meta_cols = {"day", "time", "region", "Unnamed: 0"}
        common_cols = sorted(
            list(set(pdf_val.columns) & set(pdf_req.columns) - meta_cols)
        )

        vals = pdf_val[common_cols].to_numpy().flatten()
        cnts = pdf_req[common_cols].to_numpy().flatten()

        mask = ~np.isnan(vals) & ~np.isnan(cnts) & (cnts > 0)
        return pd.DataFrame({"size": vals[mask], "count": cnts[mask]})

    meta = pd.DataFrame({"size": [], "count": []}, dtype="float64")
    return dd.map_partitions(flatten, df_val, df_req, meta=meta)


def io_transformer(dataset: str) -> Callable:
    logging.info(f"Calling io_transformer for dataset: {dataset}")
    mapping = {
        "globus24_request_body_size": _globus24_transform,
        "huawei25_request_body_size": _huawei25_transform,
        "huawei25_read_body_latency": _huawei25_transform,
        "huawei25_write_rsp_latency": _huawei25_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
