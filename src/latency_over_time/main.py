import logging

import pandas as pd

from src.latency_over_time.analysis import (
    average_latency_per_hour_csv,
    plot_duration_per_hour,
    plot_duration_per_hour_all_subplots,
)
from src.latency_over_time.data_loaders import read_data
from src.latency_over_time.transformers import (
    latency_over_time_transformer,
)
from src.util import log_timing, setup_logging, dask_client, calc_stats

setup_logging()

DATASETS = [
    "azure21",
    "globus24",
    "globus24_total",
    "huawei23",
    "huawei23_total",
    "huawei25",
    "huawei25_total",
    "ibm26",
    "ibm26_total",
]


@log_timing
@dask_client
def main():
    logging.info("Starting to load and process data...")

    for i, dataset in enumerate(DATASETS):
        logging.info(f"Loading dataset: {dataset}")
        df = read_data(dataset, latency_over_time_transformer(dataset))

        logging.info(f"Calling average_latency_per_hour for dataset: {dataset}")
        average_latency_per_hour_csv(df.copy(), dataset)


def generate_stats():
    logging.info("Generating statistics for latency over time...")
    stats_data = {}
    for dataset in DATASETS:
        for suffix in ["", "_weekdays", "_weekends"]:
            try:
                df = pd.read_csv(f"results/latency_over_time/average_latency_per_hour/csv/{dataset}{suffix}.csv")
                stats_data[f"{dataset}{suffix}"] = df["duration"].to_numpy()
            except FileNotFoundError:
                pass
    if stats_data:
        calc_stats(stats_data, "results/latency_over_time/stats/average_latency_per_hour_stats.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    for dataset in DATASETS:
        plot_duration_per_hour(dataset)
        plot_duration_per_hour(f"{dataset}_weekdays")
        plot_duration_per_hour(f"{dataset}_weekends")
        plot_duration_per_hour_all_subplots(dataset)
