import dask.dataframe as dd


def _alibaba21_memory_limit_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.groupby("functionName").mean().reset_index()
    df = df.drop(columns=["functionName"])
    df = df.rename(columns={"memoryMB": "memory_limit_mb"})
    df = df.dropna()
    df = df[df["memory_limit_mb"] != 0]
    return df


def _azure19_memory_usage_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.groupby("HashApp")["AverageAllocatedMb"].mean().reset_index()
    df = df.rename(columns={"AverageAllocatedMb": "memory_usage_mb", "HashApp": "pool"})
    df = df.dropna()
    df = df[df["memory_usage_mb"] > 0]
    return df


def _huawei23_memory_usage_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.drop(columns=["day", "time"])
    df = df.mean().reset_index()
    df = df.rename(columns={"index": "pool", 0: "memory_usage_fraction"})
    df = df.dropna()
    df = df[df["memory_usage_fraction"] != 0]
    return df


def _huawei23_memory_limit_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.drop(columns=["day", "time"])
    df = df.mean().reset_index()
    df = df.rename(columns={"index": "pool", 0: "memory_limit_mb"})
    df = df.dropna()
    df = df[df["memory_limit_mb"] != 0]
    return df


def _huawei23_cpu_limit_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.drop(columns=["day", "time"])
    df = df.mean().reset_index()
    df = df.rename(columns={"index": "pool", 0: "cpu_limit_cores"})
    df = df.dropna()
    df = df[df["cpu_limit_cores"] != 0]
    return df


def _huawei23_cpu_usage_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.drop(columns=["day", "time"])
    df = df.mean().reset_index()
    df = df.rename(columns={"index": "pool", 0: "cpu_usage_fraction"})
    df = df.dropna()
    df = df[df["cpu_usage_fraction"] != 0]
    return df


def _huawei25_memory_usage_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.rename(columns={"value": "memory_usage_mb"})
    df["pool"] = df["pool"].astype("string")
    df["pool_memory"] = (
        df["pool"].str.extract(r"-(\d+)$", expand=False).astype("float64")
    )
    df["memory_usage_fraction"] = df["memory_usage_mb"] / df["pool_memory"]
    df = df.dropna()
    return df[["pool", "memory_usage_fraction", "memory_usage_mb", "pool_memory"]]


def _huawei25_memory_limit_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["pool"] = df["pool"].astype("string")
    df["memory_limit_mb"] = (
        df["pool"].str.extract(r"-(\d+)$", expand=False).astype("float64")
    )
    df = df[["pool", "memory_limit_mb"]].drop_duplicates()
    df = df.dropna()
    return df


def _huawei25_cpu_usage_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.rename(columns={"value": "cpu_usage_cores"})
    df["pool"] = df["pool"].astype("string")
    df["pool_cpu"] = (
        df["pool"].str.extract(r"-(\d+)-[^-]+$", expand=False).astype("float64")
    )
    df["cpu_usage_fraction"] = (df["cpu_usage_cores"] * 1000) / df["pool_cpu"]
    df = df.dropna()
    return df[["pool", "cpu_usage_fraction", "cpu_usage_cores", "pool_cpu"]]


def _huawei25_cpu_limit_transform(df: dd.DataFrame) -> dd.DataFrame:
    df["pool"] = df["pool"].astype("string")
    df["cpu_limit_cores"] = (
        df["pool"].str.extract(r"-(\d+)-[^-]+$", expand=False).astype("float64")
    )
    df["cpu_limit_cores"] = df["cpu_limit_cores"] / 1000
    df = df[["pool", "cpu_limit_cores"]].drop_duplicates()
    df = df.dropna()
    return df


def _ibm26_memory_limit_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna()
    df = df[df["memory_limit_mb"] > 0]
    return df


def _ibm26_cpu_limit_transform(df: dd.DataFrame) -> dd.DataFrame:
    df = df.dropna()
    df = df[df["cpu_limit_cores"] > 0]
    return df


def resource_usage_transformer(dataset: str):
    mapping = {
        "alibaba21_memory_limit": _alibaba21_memory_limit_transform,
        "azure19_memory_usage": _azure19_memory_usage_transform,
        "huawei23_memory_usage": _huawei23_memory_usage_transform,
        "huawei23_memory_limit": _huawei23_memory_limit_transform,
        "huawei23_cpu_limit": _huawei23_cpu_limit_transform,
        "huawei23_cpu_usage": _huawei23_cpu_usage_transform,
        "huawei25_memory_usage": _huawei25_memory_usage_transform,
        "huawei25_memory_limit": _huawei25_memory_limit_transform,
        "huawei25_cpu_usage": _huawei25_cpu_usage_transform,
        "huawei25_cpu_limit": _huawei25_cpu_limit_transform,
        "ibm26_memory_limit": _ibm26_memory_limit_transform,
        "ibm26_cpu_limit": _ibm26_cpu_limit_transform,
    }
    transformer = mapping.get(dataset)
    if transformer is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    return transformer
