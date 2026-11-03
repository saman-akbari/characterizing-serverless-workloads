import pandas as pd

from src.local import data_path


def load_functions() -> pd.DataFrame:
    df = pd.read_csv(
        f"{data_path}/globus24/GlobusComputeDataset-v1-functions.csv",
        usecols=["function_uuid", "num_of_imports", "loc", "cyc_complexity"],
    )
    return df


def load_tasks() -> pd.DataFrame:
    df = pd.read_csv(
        f"{data_path}/globus24/GlobusComputeDataset-v1-tasks.csv",
        usecols=["function_uuid", "received", "result_received"],
    )
    return df
