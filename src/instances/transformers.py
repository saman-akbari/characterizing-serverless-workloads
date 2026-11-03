import logging
from typing import Callable

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def _huawei_scaling_transform(df: dd.DataFrame) -> dd.DataFrame:
    def process_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        meta_cols = ["day", "time", "region", "Unnamed: 0"]
        func_cols = [c for c in pdf.columns if c not in meta_cols]
        data_df = pdf[func_cols].fillna(0)

        local_max = data_df.max()

        diffs = data_df.diff()
        local_scale_up = diffs.clip(lower=0).max()
        local_scale_down = diffs.clip(upper=0).abs().max()

        return pd.DataFrame(
            {
                "max_instances": local_max,
                "scale_up_rate": local_scale_up,
                "scale_down_rate": local_scale_down,
            }
        )

    meta_df = pd.DataFrame(
        {
            "max_instances": pd.Series(dtype="float"),
            "scale_up_rate": pd.Series(dtype="float"),
            "scale_down_rate": pd.Series(dtype="float"),
        }
    )

    per_day_stats = df.map_partitions(process_partition, meta=meta_df)
    final_stats = per_day_stats.groupby(per_day_stats.index).max()
    final_stats = (
        final_stats.compute().reset_index().rename(columns={"index": "func_id"})
    )
    final_stats = final_stats[final_stats["max_instances"] > 0]
    return dd.from_pandas(final_stats, npartitions=1)


def _huawei_lifespan_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["podID"])

    trace_min, trace_max = dask.compute(
        df["time_worker"].min(), df["time_worker"].max()
    )
    agg = df.groupby("podID")["time_worker"].agg(["min", "max"])
    agg = agg[(agg["min"] > trace_min) & (agg["max"] < trace_max)]

    agg["lifespan"] = agg["max"] - agg["min"]
    return agg[["lifespan"]].reset_index()


def _ibm26_scaling_transform(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    valid_rows = df[df["InvocationTimes"].map(len) > 0]

    for row in valid_rows.itertuples():
        ts = np.array(row.InvocationTimes)
        pods = np.array(row.PodHash)

        df_func = pd.DataFrame({"ts": ts, "pod": pods}).drop_duplicates()

        per_sec_counts = df_func["ts"].value_counts()
        minutes = (per_sec_counts.index // 60).astype(int)
        concurrency = per_sec_counts.groupby(minutes).max()

        full_idx = range(concurrency.index.min(), concurrency.index.max() + 1)
        concurrency = concurrency.reindex(full_idx, fill_value=0)

        results.append(
            {
                "func_id": row.AppHash,
                "max_instances": concurrency.max(),
                "scale_up_rate": concurrency.diff().max(),
                "scale_down_rate": concurrency.diff().clip(upper=0).abs().max(),
            }
        )

    return pd.DataFrame(results).fillna(0)


def _ibm26_lifespan_transform(df: pd.DataFrame) -> pd.DataFrame:
    valid_rows = df[df["InvocationTimes"].map(len) > 0]

    all_times = np.concatenate(valid_rows["InvocationTimes"].values)
    all_pods = np.concatenate(valid_rows["PodHash"].values)

    window_min = all_times.min()
    window_max = all_times.max()

    tmp = pd.DataFrame({"p": all_pods, "t": all_times})
    final = tmp.groupby("p")["t"].agg(min="min", max="max").reset_index()

    final = final[(final["min"] > window_min) & (final["max"] < window_max)].copy()

    final.columns = ["pod_hash", "min_timestamp", "max_timestamp"]
    final["lifespan"] = final["max_timestamp"] - final["min_timestamp"]
    return final


def _ibm26_config_concurrency(df: pd.DataFrame) -> pd.DataFrame:
    all_vals = [
        x for lst in df["AppContainerConcurrency"] for x in lst if not np.isnan(x)
    ]
    return pd.DataFrame({"concurrency_limit": all_vals})


def _ibm26_config_min_scale(df: pd.DataFrame) -> pd.DataFrame:
    all_vals = [x for lst in df["AppMinScale"] for x in lst if not np.isnan(x)]
    return pd.DataFrame({"min_scale": all_vals})


def instances_transformer(dataset: str) -> Callable:
    logging.info(f"Calling instances_transformer for dataset: {dataset}")
    mapping = {
        "huawei23": _huawei_scaling_transform,
        "huawei25": _huawei_scaling_transform,
        "huawei25_lifespan": _huawei_lifespan_transform,
        "ibm26": _ibm26_scaling_transform,
        "ibm26_lifespan": _ibm26_lifespan_transform,
        "ibm26_concurrency_limit": _ibm26_config_concurrency,
        "ibm26_min_scale": _ibm26_config_min_scale,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
