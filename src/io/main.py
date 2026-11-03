import logging

import pandas as pd

from src.io.analysis import (
    compute_cdf,
    plot_request_body_size_cdf,
    plot_io_latency_cdf,
    plot_huawei25_runtime_io_latency_cdf,
)
from src.io.data_loaders import read_data
from src.io.transformers import io_transformer
from src.util import (
    log_timing,
    setup_logging,
    dask_client,
    calc_stats,
    to_csv_safe_dask,
)

setup_logging()

DATASETS = [
    "globus24_request_body_size",
    "huawei25_request_body_size",
    "huawei25_read_body_latency",
    "huawei25_write_rsp_latency",
]


@log_timing
@dask_client
def main():
    logging.info("Starting to load and process I/O data...")

    for dataset in DATASETS:
        logging.info(f"Loading dataset: {dataset}")
        df = read_data(dataset, io_transformer(dataset))
        to_csv_safe_dask(df, f"results/io/csv/{dataset}.csv")

        logging.info(f"Processing CDF for dataset: {dataset}")
        compute_cdf(dataset, df.copy())


def generate_stats():
    logging.info("Generating statistics for I/O...")
    size_stats = {}
    io_latency_stats = {}
    for dataset in DATASETS:
        df_pd = pd.read_csv(f"results/io/csv/{dataset}.csv")
        if "count" in df_pd.columns:
            values = df_pd["size"].to_numpy()
            weights = df_pd["count"].to_numpy()
            if "latency" in dataset:
                io_latency_stats[dataset] = (values, weights)
            else:
                size_stats[dataset] = (values / 1024, weights)
        else:
            if "latency" in dataset:
                io_latency_stats[dataset] = df_pd["size"].to_numpy()
            else:
                size_stats[dataset] = df_pd["size"].to_numpy() / 1024
    if size_stats:
        calc_stats(size_stats, "results/io/stats/request_body_size.csv")
    if io_latency_stats:
        calc_stats(io_latency_stats, "results/io/stats/io_latency.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    plot_request_body_size_cdf(
        ["globus24_request_body_size", "huawei25_request_body_size"]
    )
    plot_io_latency_cdf(["huawei25_read_body_latency", "huawei25_write_rsp_latency"])
    plot_huawei25_runtime_io_latency_cdf()
