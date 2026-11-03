import logging

import pandas as pd

from src.latency_over_time.analysis import (
    average_latency_per_hour_csv as save_average_latency_csv,
)
from src.latency_over_time.data_loaders import read_data as read_duration_data
from src.latency_over_time.transformers import latency_over_time_transformer
from src.request_patterns.analysis import (
    average_requests_per_hour_csv,
    plot_request_patterns_time_series,
    plot_load_over_time,
    save_arrival_time_rates,
    save_load_over_time_data,
    plot_requests_per_hour,
    save_iat_cdf,
    plot_iat_cdf,
    save_function_popularity_cdf,
    plot_function_popularity_cdf,
)
from src.request_patterns.data_loaders import read_data
from src.request_patterns.transformers import (
    request_patterns_transformer,
    iat_transformer,
    popularity_transformer,
)
from src.util import log_timing, setup_logging, dask_client, calc_stats

setup_logging()

REQUEST_PATTERNS_DATASETS = [
    "azure19",
    "azure21",
    "globus24",
    "huawei23",
    "huawei25",
    "ibm26",
]

IDLE_TIME_DATASETS = [
    "azure21",
    "globus24",
    "huawei23",
    "ibm26",
]

POPULARITY_DATASETS = [
    "azure19",
    "azure21",
    "globus24",
    "huawei23",
    "huawei25",
    "ibm26",
]

LOAD_DATASETS = [
    "azure21",
    "globus24",
    "huawei23",
    "huawei25",
    "ibm26",
]


@log_timing
@dask_client
def main():
    logging.info("Starting to load and process data...")

    for i, dataset in enumerate(REQUEST_PATTERNS_DATASETS):
        logging.info(f"Loading dataset for request patterns: {dataset}")
        df = read_data(dataset, request_patterns_transformer(dataset))

        logging.info(
            f"Calculating request_patterns/average_requests_per_hour for dataset: {dataset}"
        )
        average_requests_per_hour_csv(df.copy(), dataset)

        logging.info(f"Calculating request_patterns/time_series for dataset: {dataset}")
        save_arrival_time_rates(df.copy(), dataset)

        if dataset in LOAD_DATASETS:
            logging.info(
                f"Calculating latency_over_time/average_latency_per_hour for dataset: {dataset}"
            )
            df_duration = read_duration_data(
                dataset, latency_over_time_transformer(dataset)
            )
            save_average_latency_csv(df_duration.copy(), dataset)
            save_load_over_time_data(dataset)

    logging.info("Starting Idle Time analysis...")
    for dataset in IDLE_TIME_DATASETS:
        logging.info(f"Loading dataset for IAT: {dataset}")
        df_iat = read_data(dataset, iat_transformer(dataset))
        logging.info(f"Calculating IAT CDF for dataset: {dataset}")
        save_iat_cdf(df_iat, dataset)

    logging.info("Starting Function Popularity analysis...")
    for dataset in POPULARITY_DATASETS:
        logging.info(f"Loading dataset for Function Popularity: {dataset}")
        df_pop = read_data(dataset, popularity_transformer(dataset))
        logging.info(f"Calculating Function Popularity CDF for dataset: {dataset}")
        save_function_popularity_cdf(df_pop, dataset)


def generate_stats():
    logging.info("Generating statistics for request patterns...")
    request_rate_stats = {}
    for dataset in REQUEST_PATTERNS_DATASETS:
        df_pd = pd.read_csv(f"results/request_patterns/time_series/csv/{dataset}.csv")
        request_rate_stats[dataset] = df_pd["requests"].to_numpy()
    calc_stats(request_rate_stats, "results/request_patterns/stats/requests_per_minute_stats.csv")

    avg_req_stats = {}
    for dataset in REQUEST_PATTERNS_DATASETS:
        for suffix in ["", "_weekdays", "_weekends"]:
            try:
                df = pd.read_csv(f"results/request_patterns/average_requests_per_hour/csv/{dataset}{suffix}.csv")
                avg_req_stats[f"{dataset}{suffix}"] = df["requests"].to_numpy()
            except FileNotFoundError:
                pass
    if avg_req_stats:
        calc_stats(avg_req_stats, "results/request_patterns/stats/average_requests_per_hour_stats.csv")

    load_stats = {}
    for dataset in LOAD_DATASETS:
        df = pd.read_csv(f"results/request_patterns/load_over_time/csv/{dataset}.csv")
        load_stats[dataset] = df["load"].to_numpy()
    if load_stats:
        calc_stats(load_stats, "results/request_patterns/stats/load_over_time_stats.csv")

    popularity_stats = {}
    for dataset in POPULARITY_DATASETS:
        df = pd.read_csv(f"results/request_patterns/function_popularity/csv/{dataset}.csv")
        popularity_stats[dataset] = df["request_percentage"].to_numpy()
    if popularity_stats:
        calc_stats(popularity_stats, "results/request_patterns/stats/function_popularity_stats.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    for dataset in REQUEST_PATTERNS_DATASETS:
        plot_request_patterns_time_series(dataset)
        plot_requests_per_hour(dataset)
    plot_load_over_time(LOAD_DATASETS)
    plot_iat_cdf(IDLE_TIME_DATASETS)
    plot_function_popularity_cdf(POPULARITY_DATASETS)
