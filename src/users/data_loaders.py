import gc
import glob

import dask.dataframe as dd
import pandas as pd

from src.local import data_path


def _read_azure19() -> dd.DataFrame:
    file_pattern = f"{data_path}/azure19/invocations_per_function_md.anon.*.csv"
    minute_cols = [str(i) for i in range(1, 1441)]
    usecols = ["HashOwner", "HashFunction"] + minute_cols
    dtype_dict = {col: "Int64" for col in minute_cols}

    return dd.read_csv(
        file_pattern,
        blocksize="16MB",
        assume_missing=True,
        usecols=usecols,
        dtype=dtype_dict,
    )


def _read_azure20() -> dd.DataFrame:
    file_pattern = f"{data_path}/azure20/azurefunctions-accesses-2020.csv"
    return dd.read_csv(
        file_pattern,
        blocksize="64MB",
        assume_missing=True,
        usecols=["AnonUserId", "AnonAppName", "AnonFunctionInvocationId"],
    )


def _read_globus24() -> dd.DataFrame:
    file_pattern = f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv"
    return dd.read_csv(
        file_pattern,
        blocksize="64MB",
        assume_missing=True,
        usecols=["anonymized_user_uuid", "function_uuid", "task_uuid"],
    )


def _read_huawei25() -> dd.DataFrame:
    dfs = []
    for r in range(1, 6):
        files = sorted(
            glob.glob(f"{data_path}/huawei25/time_series/R{r}/requests/*.csv")
        )

        d = dd.read_csv(
            files,
            dtype={"day": "Int64", "time": "Int64"},
            blocksize="64MB",
            encoding="utf-8",
            assume_missing=True,
        )

        func_cols = [c for c in d.columns if "---" in c]
        d = d[func_cols]

        requests_series = d.sum(axis=0)
        d_agg = requests_series.to_frame(name="requests")
        d_agg = d_agg.reset_index().rename(columns={"index": "function_id"})
        dfs.append(d_agg)

    df = dd.concat(dfs, axis=0)
    df = df.groupby("function_id")["requests"].sum().reset_index()
    df = df[df["requests"] > 0]

    def extract_user_id(partition):
        partition["function_id"] = partition["function_id"].astype(str)
        split_parts = partition["function_id"].str.split("---", expand=True)
        partition["user_id"] = split_parts[1]
        return partition[["user_id", "function_id", "requests"]]

    meta = pd.DataFrame(
        {
            "user_id": pd.Series(dtype="str"),
            "function_id": pd.Series(dtype="str"),
            "requests": pd.Series(dtype="float"),
        }
    )

    return df.map_partitions(extract_user_id, meta=meta)


def _read_ibm26() -> dd.DataFrame:
    files = glob.glob(f"{data_path}/ibm26/week_*_processed.pickle")

    dfs = []
    for file in files:
        df = pd.read_pickle(file)
        dfs.append(df[["NamespaceHash", "AppHash", "NumEvents"]].copy())

        del df
        gc.collect()

    if not dfs:
        meta = pd.DataFrame(
            {
                "NamespaceHash": pd.Series(dtype="str"),
                "AppHash": pd.Series(dtype="str"),
                "NumEvents": pd.Series(dtype="int"),
            }
        )
        return dd.from_pandas(meta, npartitions=1)

    combined = pd.concat(dfs, ignore_index=True)
    return dd.from_pandas(combined, npartitions=1)


def read_data(dataset: str) -> dd.DataFrame:
    dataset_readers = {
        "azure19": _read_azure19,
        "azure20": _read_azure20,
        "globus24": _read_globus24,
        "huawei25": _read_huawei25,
        "ibm26": _read_ibm26,
    }
    reader = dataset_readers.get(dataset)
    if reader is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return reader()
