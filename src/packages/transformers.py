import pandas as pd


def globus24_functions_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["num_of_imports", "loc", "cyc_complexity"])
    df = df.rename(
        columns={
            "num_of_imports": "imports",
            "loc": "loc",
            "cyc_complexity": "complexity",
        }
    )
    return df[["function_uuid", "imports", "loc", "complexity"]]


def globus24_tasks_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["received", "result_received"]).copy()
    df["latency"] = df["result_received"] - df["received"]
    return df[["function_uuid", "latency"]]


def merge_data(functions: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        tasks,
        functions,
        on="function_uuid",
        how="inner",
    )
    df = df.drop("function_uuid", axis=1)
    return df
