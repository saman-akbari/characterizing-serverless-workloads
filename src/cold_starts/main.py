import logging

import pandas as pd

from src.cold_starts.analysis import (
    cold_starts_cdf,
    cold_starts_cdf_to_csv,
    cold_starts_violin,
    cold_starts_violin_to_csv,
    cold_starts_breakdown,
    cold_starts_breakdown_to_csv,
    cold_starts_rate_to_csv,
    cold_starts_rate,
)
from src.cold_starts.data_loaders import read_data
from src.cold_starts.transformers import cold_starts_transformer
from src.util import log_timing, setup_logging, dask_client, calc_stats

setup_logging()

LATENCY_CDF_DATASETS = [
    "alibaba21",
    "globus24",
    "huawei25"
]

VIOLIN_DATASETS = [
    "alibaba21",
    "huawei25_with_runtime",
]

BREAKDOWN_DATASETS = [
    "huawei25_breakdown",
]

RATE_DATASETS = [
    "globus24_rate",
    "huawei25_rate",
    "ibm26_rate",
]


@log_timing
@dask_client
def main():
    logging.info("Starting to load and process data...")

    for dataset in LATENCY_CDF_DATASETS:
        df = cold_starts_transformer(dataset)(read_data(dataset))
        cold_starts_cdf_to_csv(df.copy(), dataset)

    for dataset in VIOLIN_DATASETS:
        df = cold_starts_transformer(dataset)(read_data(dataset))
        cold_starts_violin_to_csv(df.copy(), dataset.replace("_with_runtime", ""))
        logging.info(f"Processed Violin for: {dataset}")

    for dataset in BREAKDOWN_DATASETS:
        df = cold_starts_transformer(dataset)(read_data(dataset))
        cold_starts_breakdown_to_csv(df.copy(), dataset)
        logging.info(f"Processed Breakdown for: {dataset}")

    for dataset in RATE_DATASETS:
        df = cold_starts_transformer(dataset)(read_data(dataset))
        cold_starts_rate_to_csv(df.copy(), dataset)
        logging.info(f"Processed Rate for: {dataset}")


def generate_stats():
    logging.info("Generating statistics for cold starts...")
    stats_data = {}
    for dataset in LATENCY_CDF_DATASETS:
        df = pd.read_csv(f"results/cold_starts/cold_starts_cdf/csv/{dataset}.csv")
        stats_data[dataset] = df["cold_start"].to_numpy()
    calc_stats(stats_data, "results/cold_starts/stats/cold_start_latency.csv")

    stats_breakdown = {}
    for dataset in BREAKDOWN_DATASETS:
        df = pd.read_csv(f"results/cold_starts/breakdown/csv/{dataset}.csv")
        cols = [
            "podAllocationCost",
            "deployCodeCost",
            "deployDependencyCost",
            "schedulingCost",
        ]
        for col in cols:
            stats_breakdown[f"{dataset}_{col}"] = df[col].to_numpy()
    calc_stats(stats_breakdown, "results/cold_starts/stats/cold_start_breakdown.csv")

    stats_rate = {}
    for dataset in RATE_DATASETS:
        df = pd.read_csv(f"results/cold_starts/rate/csv/{dataset}.csv")
        rate_col = "rate_req" if "rate_req" in df.columns else "rate"
        stats_rate[f"{dataset}_req"] = df[rate_col].to_numpy()
        if "rate_func" in df.columns:
            stats_rate[f"{dataset}_func"] = df["rate_func"].to_numpy()
    calc_stats(stats_rate, "results/cold_starts/stats/cold_start_rate.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    logging.info("Plotting cold start latency CDFs...")
    cold_starts_cdf(LATENCY_CDF_DATASETS)

    logging.info("Plotting cold start latency by runtime violins...")
    for dataset in VIOLIN_DATASETS:
        cold_starts_violin(dataset.replace("_with_runtime", ""))

    logging.info("Plotting cold start breakdown...")
    for dataset in BREAKDOWN_DATASETS:
        cold_starts_breakdown(dataset)

    logging.info("Plotting cold start rate...")
    for dataset in RATE_DATASETS:
        cold_starts_rate(dataset)
