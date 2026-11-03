import glob
import re

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from src.local import data_path


def _read_alibaba21() -> dd.DataFrame:
    file_pattern = f"{data_path}/alibaba21/*.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["latency", "runtime"],
    )
    return df


def _read_huawei25() -> dd.DataFrame:
    file_pattern = f"{data_path}/huawei25/cold_starts/R*/*.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["totalCost_cold_start"],
    )
    return df


def _read_huawei25_with_runtime() -> dd.DataFrame:
    file_pattern = f"{data_path}/huawei25/cold_starts/R2/*.csv"

    df_cold_starts = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["totalCost_cold_start", "funcName", "userID", "podID"],
    )
    df_cold_starts = df_cold_starts.dropna()
    df_cold_starts["poolName"] = df_cold_starts["podID"].str.rsplit("-", n=1).str[0]
    df_cold_starts["funcID"] = (
            df_cold_starts["funcName"].astype(int).astype(str)
            + "---"
            + df_cold_starts["userID"].astype(int).astype(str)
            + "---"
            + df_cold_starts["poolName"].astype(str)
    )
    df_cold_starts = df_cold_starts.drop(
        ["funcName", "userID", "podID", "poolName"], axis=1
    )
    df_runtime = dd.read_csv(
        f"{data_path}/huawei25/trigger_types/R2/*.csv",
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["funcID", "runtime"],
    )
    df_cold_starts["funcID"] = df_cold_starts["funcID"].astype(str)
    df_runtime["funcID"] = df_runtime["funcID"].astype(str)

    df_cold_starts = df_cold_starts.set_index("funcID", sorted=False)
    df_runtime = df_runtime.set_index("funcID", sorted=False)

    df = dd.merge(df_cold_starts, df_runtime, on="funcID", how="inner")
    df = df.reset_index()

    return df


def _read_huawei25_breakdown() -> dd.DataFrame:
    file_pattern = f"{data_path}/huawei25/cold_starts/R*/*.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=[
            "day",
            "podAllocationCost",
            "deployCodeCost",
            "deployDependencyCost",
            "schedulingCost",
        ],
    )
    return df


def _read_huawei25_rate() -> dd.DataFrame:
    @dask.delayed
    def process_request_file(path):
        df = pd.read_csv(path)
        if len(df) == 0:
            return -1, {}
        day = df["day"].iloc[0]
        reqs = df.drop(columns=["day", "time", "region"], errors="ignore").sum()
        return day, reqs.to_dict()

    @dask.delayed
    def process_cold_start_file(path):
        df = pd.read_csv(path, usecols=["day", "podID", "funcName", "userID"])
        if len(df) == 0:
            return -1, {}
        day = df["day"].iloc[0]
        df = df.dropna(subset=["podID", "funcName", "userID"])
        df["poolName"] = df["podID"].str.rsplit("-", n=1).str[0]
        df["funcID"] = (
                df["funcName"].astype(int).astype(str)
                + "---"
                + df["userID"].astype(int).astype(str)
                + "---"
                + df["poolName"].astype(str)
        )
        cs = df.groupby("funcID").size()
        return day, cs.to_dict()

    cs_files = glob.glob(f"{data_path}/huawei25/cold_starts/R*/*.csv")
    cs_results = dask.compute(*[process_cold_start_file(f) for f in cs_files])
    cs_map = {}
    for day, count_dict in cs_results:
        if day == -1:
            continue
        if day not in cs_map:
            cs_map[day] = {}
        for k, v in count_dict.items():
            cs_map[day][k] = cs_map[day].get(k, 0) + v

    req_files = glob.glob(f"{data_path}/huawei25/time_series/R*/requests/*.csv")
    req_results = dask.compute(*[process_request_file(f) for f in req_files])
    req_map = {}
    for day, count_dict in req_results:
        if day == -1:
            continue
        if day not in req_map:
            req_map[day] = {}
        for k, v in count_dict.items():
            req_map[day][k] = req_map[day].get(k, 0) + v

    all_days = sorted(list(set(cs_map.keys()) | set(req_map.keys())))
    data = []
    for d in all_days:
        reqs_d = req_map.get(d, {})
        cs_d = cs_map.get(d, {})

        total_cs = sum(cs_d.values())
        total_reqs = sum(reqs_d.values())

        func_rates = []
        all_funcs = set(reqs_d.keys()) | set(cs_d.keys())
        for f in all_funcs:
            r = reqs_d.get(f, 0)
            c = cs_d.get(f, 0)
            if r > 0:
                func_rates.append(c / r)
            elif c > 0:
                func_rates.append(1.0)

        rate_func = np.mean(func_rates) if func_rates else 0.0

        data.append({
            "day": d,
            "cold_starts": total_cs,
            "requests": total_reqs,
            "rate_func": rate_func
        })

    pdf = pd.DataFrame(data)
    return dd.from_pandas(pdf, npartitions=1)


# cold start occurs when a request is served by a newly created pod
def _read_ibm26_rate() -> dd.DataFrame:
    files = glob.glob(f"{data_path}/ibm26/week_*_processed.pickle")
    file_list = sorted(
        files, key=lambda f: int(re.search(r"week_(\d+)_processed", f).group(1))
    )
    data = []

    # Load config for AppMinScale
    config_df = pd.read_pickle(f"{data_path}/ibm26/app_configs_processed.pickle")
    min_scale_map = dict(
        zip(
            config_df["AppHash"],
            config_df["AppMinScale"].apply(
                lambda x: x[0] if isinstance(x, list) else x
            ),
        )
    )

    # Track seen pods per app to compare against min_scale
    app_seen_pods = {}

    for file in file_list:
        df = pd.read_pickle(file)

        for _, row in df.iterrows():
            app = row["AppHash"]
            times = np.array(row["InvocationTimes"])
            if len(times) == 0:
                continue

            # Subtract 3-day offset applied in preprocessor
            days = ((times - 259200) // 86400).astype(int)
            unique_days, req_counts = np.unique(days, return_counts=True)
            day_reqs = dict(zip(unique_days, req_counts))

            current_pods = app_seen_pods.setdefault(app, set())
            min_s = min_scale_map.get(app, 0)
            cold_starts_per_day = {}

            for i in np.argsort(times, kind='stable'):
                pod = row["PodHash"][i]

                # Check if this is a new pod instance
                if pod not in current_pods:
                    current_pods.add(pod)
                    # Only count as cold start if we exceeded the pre-warmed buffer
                    if len(current_pods) > min_s:
                        day = days[i]
                        cold_starts_per_day[day] = cold_starts_per_day.get(day, 0) + 1

            # Aggregate per day for this app
            for day, reqs in day_reqs.items():
                cs = cold_starts_per_day.get(day, 0)
                data.append(
                    {
                        "app": app,
                        "day": day,
                        "cold_starts": cs,
                        "requests": reqs,
                    }
                )

    app_df = pd.DataFrame(data)
    if not app_df.empty:
        app_df["func_rate"] = app_df["cold_starts"] / app_df["requests"]

        pdf = app_df.groupby("day").agg(
            cold_starts=("cold_starts", "sum"),
            requests=("requests", "sum"),
            rate_func=("func_rate", "mean")
        ).reset_index()
    else:
        pdf = pd.DataFrame(columns=["day", "cold_starts", "requests", "rate_func"])

    return dd.from_pandas(pdf, npartitions=1)


def _read_globus24() -> dd.DataFrame:
    file_pattern = f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["execution_start", "waiting_for_nodes"],
    )
    return df


def _read_globus24_rate() -> dd.DataFrame:
    file_pattern = f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv"
    df = dd.read_csv(
        file_pattern,
        blocksize="64MB",
        encoding="utf-8",
        assume_missing=True,
        usecols=["received", "waiting_for_nodes", "function_uuid"],
    )
    df = df.dropna(subset=["received"])
    df["day"] = (df["received"] // 86400).astype(int)
    df["cold_starts"] = df["waiting_for_nodes"].notnull().astype(int)
    df["requests"] = 1

    grouped = df.groupby(["day", "function_uuid"])[["cold_starts", "requests"]].sum().compute().reset_index()
    grouped["func_rate"] = grouped["cold_starts"] / grouped["requests"]

    final_df = grouped.groupby("day").agg(
        cold_starts=("cold_starts", "sum"),
        requests=("requests", "sum"),
        rate_func=("func_rate", "mean")
    ).reset_index()

    return dd.from_pandas(final_df, npartitions=1)


def read_data(dataset: str) -> dd.DataFrame:
    dataset_readers = {
        "alibaba21": _read_alibaba21,
        "huawei25": _read_huawei25,
        "huawei25_with_runtime": _read_huawei25_with_runtime,
        "huawei25_breakdown": _read_huawei25_breakdown,
        "huawei25_rate": _read_huawei25_rate,
        "ibm26_rate": _read_ibm26_rate,
        "globus24": _read_globus24,
        "globus24_rate": _read_globus24_rate,
    }
    reader = dataset_readers.get(dataset)
    if reader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return reader()
