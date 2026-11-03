import glob
import logging
from typing import Callable, List

import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.local import data_path

logging.basicConfig(level=logging.INFO)


def _azure19_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.rename(columns={"Average": "duration", "Count": "count"})
    df = df.dropna(subset=["duration"])
    return df


def _azure19_func_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["total"] = df["Average"] * df["Count"]
    g = df.groupby("HashFunction")[["total", "Count"]].sum()
    g["duration"] = g["total"] / g["Count"]
    return g.reset_index()[["duration"]]


def _azure21_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["duration"] = df["duration"] * 1000
    return df


def _azure21_func_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["duration"] = df["duration"] * 1000
    g = df.groupby("func")["duration"].mean()
    return g.to_frame("duration").reset_index()[["duration"]]


def _globus24_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["duration"] = (df["execution_end"] - df["execution_start"]) * 1000
    df = df.drop(columns=["execution_start", "execution_end"])
    return df


def _globus24_func_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["duration"] = (df["execution_end"] - df["execution_start"]) * 1000
    g = df.groupby("function_uuid")["duration"].mean()
    return g.to_frame("duration").reset_index()[["duration"]]


def _globus24_total_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["duration"] = (df["result_received"] - df["received"]) * 1000
    df = df.drop(columns=["received", "result_received"])
    return df


def _globus24_overhead_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["duration"] = (df["execution_start"] - df["received"]) * 1000
    df = df.drop(columns=["received", "execution_start"])
    df = df.dropna(subset=["duration"])
    return df


def _globus24_infra_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["duration"] = (df["execution_end"] - df["execution_start"]) * 1000
    mapping = {
        "LocalProvider": "Local",
        "ThreadPoolEngine": "Local",
        "KubernetesProvider": "Cloud",
        "GlobusComputeEngine": "Cloud",
        "SlurmProvider": "HPC",
        "CobaltProvider": "HPC",
        "LSFProvider": "HPC",
        "PBSProProvider": "HPC",
    }
    df = df.dropna(subset=["endpoint_type", "duration"])
    df["infra"] = df["endpoint_type"].map(mapping, meta=("endpoint_type", "object"))
    df = df.dropna(subset=["infra"])
    return df[["duration", "infra"]]


def _huawei23_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    df_val, df_cnt = dfs[0], dfs[1]

    def flatten(pdf_val, pdf_cnt):
        vals = pdf_val.iloc[:, 2:].to_numpy().flatten()
        cnts = pdf_cnt.iloc[:, 2:].to_numpy().flatten()
        mask = ~np.isnan(vals) & ~np.isnan(cnts) & (cnts > 0)
        return pd.DataFrame({"duration": vals[mask], "count": cnts[mask]})

    meta = pd.DataFrame({"duration": [], "count": []}, dtype="float64")
    return dd.map_partitions(flatten, df_val, df_cnt, meta=meta)


def _huawei23_func_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    df_val, df_cnt = dfs[0], dfs[1]

    val_cols = df_val.columns[2:]
    cnt_cols = df_cnt.columns[2:]
    common_cols = list(val_cols.intersection(cnt_cols))

    df_val_c = df_val[common_cols]
    df_cnt_c = df_cnt[common_cols]

    df_cnt_valid = df_cnt_c.mask(df_val_c.isnull(), 0)
    total_dur = (df_val_c * df_cnt_c).sum().compute()
    total_cnt = df_cnt_valid.sum().compute()

    means = total_dur / total_cnt
    means = means.replace([np.inf, -np.inf], np.nan).dropna()
    return dd.from_pandas(pd.DataFrame({"duration": means.values}), npartitions=1)


def _huawei23_total_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    p, f, r = dfs[0], dfs[1], dfs[2]

    def flatten_total(pdf_p, pdf_f, pdf_r):
        vals_p = pdf_p.iloc[:, 2:].to_numpy()
        vals_f = pdf_f.iloc[:, 2:].to_numpy()
        vals_r = pdf_r.iloc[:, 2:].to_numpy()

        vals_total = (vals_p + vals_f).flatten()
        cnts = vals_r.flatten()

        mask = ~np.isnan(vals_total) & ~np.isnan(cnts) & (cnts > 0)
        return pd.DataFrame({"duration": vals_total[mask], "count": cnts[mask]})

    meta = pd.DataFrame({"duration": [], "count": []}, dtype="float64")
    return dd.map_partitions(flatten_total, p, f, r, meta=meta)


def _huawei23_overhead_transform(dfs: List[dd.DataFrame]) -> dd.DataFrame:
    return _huawei23_transform(dfs)


def _huawei25_transform(dfs: List[List[dd.DataFrame]]) -> dd.DataFrame:
    m_list, r_list = dfs[0], dfs[1]

    def flatten(pdf_val, pdf_cnt):
        meta_cols = {"day", "time", "region", "Unnamed: 0"}
        common_cols = sorted(
            list(set(pdf_val.columns) & set(pdf_cnt.columns) - meta_cols)
        )

        vals = pdf_val[common_cols].to_numpy().flatten()
        cnts = pdf_cnt[common_cols].to_numpy().flatten()
        mask = ~np.isnan(vals) & ~np.isnan(cnts) & (cnts > 0)
        return pd.DataFrame({"duration": vals[mask] * 1000, "count": cnts[mask]})

    meta = pd.DataFrame({"duration": [], "count": []}, dtype="float64")
    region_dfs = [
        dd.map_partitions(flatten, m, r, meta=meta) for m, r in zip(m_list, r_list)
    ]
    return dd.concat(region_dfs, axis=0, interleave_partitions=True)


def _huawei25_func_transform(dfs: List[List[dd.DataFrame]]) -> dd.DataFrame:
    m_list, r_list = dfs[0], dfs[1]
    all_means = []

    for m, r in zip(m_list, r_list):
        drop_cols = ["day", "time", "region", "Unnamed: 0"]
        cols = [c for c in m.columns if c not in drop_cols and c in r.columns]

        m_c = m[cols]
        r_c = r[cols]

        r_c_valid = r_c.mask(m_c.isnull(), 0)
        total_dur = (m_c * r_c * 1000).sum().compute()
        total_cnt = r_c_valid.sum().compute()

        means = total_dur / total_cnt
        means = means.replace([np.inf, -np.inf], np.nan).dropna()
        all_means.append(means)

    full_series = pd.concat(all_means)
    return dd.from_pandas(pd.DataFrame({"duration": full_series.values}), npartitions=1)


def _huawei25_total_transform(dfs: List[List[dd.DataFrame]]) -> dd.DataFrame:
    return _huawei25_transform(dfs)


def _huawei25_overhead_transform(dfs: List[List[dd.DataFrame]]) -> dd.DataFrame:
    e_list, t_list, r_list = dfs[0], dfs[1], dfs[2]

    def flatten_overhead(pdf_e, pdf_t, pdf_r):
        meta_cols = {"day", "time", "region", "Unnamed: 0"}
        common_cols = sorted(
            list(
                set(pdf_e.columns) & set(pdf_t.columns) & set(pdf_r.columns) - meta_cols
            )
        )

        e_vals = pdf_e[common_cols].to_numpy()
        t_vals = pdf_t[common_cols].to_numpy()
        r_vals = pdf_r[common_cols].to_numpy()

        vals = (t_vals - e_vals).flatten()
        cnts = r_vals.flatten()

        mask = ~np.isnan(vals) & ~np.isnan(cnts) & (cnts > 0)
        return pd.DataFrame(
            {"duration": np.clip(vals[mask], 0, None) * 1000, "count": cnts[mask]}
        )

    meta = pd.DataFrame({"duration": [], "count": []}, dtype="float64")
    region_dfs = [
        dd.map_partitions(flatten_overhead, e, t, r, meta=meta)
        for e, t, r in zip(e_list, t_list, r_list)
    ]
    return dd.concat(region_dfs, axis=0, interleave_partitions=True)


def _huawei25_runtime_transform(dfs: List[List[dd.DataFrame]]) -> dd.DataFrame:
    m_list, r_list = dfs[0], dfs[1]

    files = glob.glob(
        f"{data_path}/huawei25/**/df_funcID_runtime_triggerType.csv", recursive=True
    )
    mapping_df = pd.read_csv(files[0])
    runtime_map = dict(zip(mapping_df["funcID"], mapping_df["runtime"]))

    def flatten_runtime(pdf_val, pdf_cnt):
        meta_cols = {"day", "time", "region", "Unnamed: 0"}
        common_cols = sorted(
            list(set(pdf_val.columns) & set(pdf_cnt.columns) - meta_cols)
        )

        vals = pdf_val[common_cols].to_numpy()
        cnts = pdf_cnt[common_cols].to_numpy()

        col_runtimes = np.array([runtime_map.get(c, "unknown") for c in common_cols])
        runtimes = np.tile(col_runtimes, (vals.shape[0], 1)).flatten()

        vals_flat = vals.flatten()
        cnts_flat = cnts.flatten()

        mask = ~np.isnan(vals_flat) & ~np.isnan(cnts_flat) & (cnts_flat > 0)
        return pd.DataFrame(
            {
                "duration": vals_flat[mask] * 1000,
                "count": cnts_flat[mask],
                "infra": runtimes[mask],
            }
        )

    meta = pd.DataFrame({"duration": [], "count": [], "infra": []}).astype(
        {"duration": "float64", "count": "float64", "infra": "object"}
    )
    region_dfs = [
        dd.map_partitions(flatten_runtime, m, r, meta=meta)
        for m, r in zip(m_list, r_list)
    ]
    return dd.concat(region_dfs, axis=0, interleave_partitions=True)


def _ibm26_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["duration"])
    return df


def _ibm26_func_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["duration"])
    return df


def _ibm26_total_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["duration"])
    return df


def _ibm26_overhead_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna(subset=["duration"])
    return df


def latency_transformer(dataset: str) -> Callable:
    logging.info(f"Calling transformer for dataset: {dataset}")
    mapping = {
        "azure19": _azure19_transform,
        "azure19_func": _azure19_func_transform,
        "azure21": _azure21_transform,
        "azure21_func": _azure21_func_transform,
        "globus24": _globus24_transform,
        "globus24_func": _globus24_func_transform,
        "globus24_total": _globus24_total_transform,
        "globus24_overhead": _globus24_overhead_transform,
        "globus24_infra": _globus24_infra_transform,
        "huawei23": _huawei23_transform,
        "huawei23_func": _huawei23_func_transform,
        "huawei23_total": _huawei23_total_transform,
        "huawei23_overhead": _huawei23_overhead_transform,
        "huawei25": _huawei25_transform,
        "huawei25_func": _huawei25_func_transform,
        "huawei25_total": _huawei25_total_transform,
        "huawei25_overhead": _huawei25_overhead_transform,
        "huawei25_runtime": _huawei25_runtime_transform,
        "ibm26": _ibm26_transform,
        "ibm26_func": _ibm26_func_transform,
        "ibm26_total": _ibm26_total_transform,
        "ibm26_overhead": _ibm26_overhead_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
