import logging

import pandas as pd

from src.users.analysis import users_functions_to_csv, users_requests_to_csv, plot_functions_per_user_cdf, plot_requests_per_user_cdf
from src.users.data_loaders import read_data
from src.users.transformers import user_transformer
from src.util import log_timing, setup_logging, dask_client, calc_stats

setup_logging()

FUNCTIONS_DATASETS = [
    "azure19",
    "azure20",
    "huawei25",
    "ibm26",
]

REQUESTS_DATASETS = [
    "azure19",
    "azure20",
    "globus24",
    "huawei25",
    "ibm26",
]


@log_timing
@dask_client
def main():
    logging.info("Starting user analysis...")

    for dataset in FUNCTIONS_DATASETS:
        df = user_transformer(dataset)(read_data(dataset))
        users_functions_to_csv(df.copy(), dataset)
        logging.info(f"Processed functions-per-user analysis for: {dataset}")

    for dataset in REQUESTS_DATASETS:
        df = user_transformer(dataset)(read_data(dataset))
        users_requests_to_csv(df.copy(), dataset)
        logging.info(f"Processed requests-per-user analysis for: {dataset}")


def generate_stats():
    logging.info("Generating user statistics...")
    stats_functions = {}
    stats_requests = {}

    for dataset in FUNCTIONS_DATASETS:
        df_func = pd.read_csv(f"results/users/functions_per_user/csv/{dataset}.csv")
        stats_functions[dataset] = df_func["value"].to_numpy()

    for dataset in REQUESTS_DATASETS:
        df_req = pd.read_csv(f"results/users/requests_per_user/csv/{dataset}.csv")
        stats_requests[dataset] = df_req["value"].to_numpy()

    calc_stats(stats_functions, "results/users/stats/functions_per_user_stats.csv")
    calc_stats(stats_requests, "results/users/stats/requests_per_user_stats.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    logging.info("Plotting user analysis CDFs...")
    plot_functions_per_user_cdf(FUNCTIONS_DATASETS)
    plot_requests_per_user_cdf(REQUESTS_DATASETS)
