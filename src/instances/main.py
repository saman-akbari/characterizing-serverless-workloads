import logging

import pandas as pd

from src.instances.analysis import (
    compute_instance_metrics_cdf,
    plot_metric_cdf,
    compute_lifespan_cdf,
    plot_lifespan_cdf,
    compute_config_cdf,
    plot_concurrency_limit_cdf,
    plot_min_scale_cdf,
)
from src.instances.data_loaders import read_data
from src.instances.transformers import instances_transformer
from src.util import log_timing, setup_logging, dask_client, calc_stats

setup_logging()

SCALE_DATASETS = [
    "huawei23",
    "huawei25",
    "ibm26"
]

LIFESPAN_DATASETS = [
    "huawei25_lifespan",
    "ibm26_lifespan"
]

CONFIG_DATASETS = [
    "ibm26_concurrency_limit",
    "ibm26_min_scale"
]


@log_timing
@dask_client
def main():
    logging.info("Starting instance/pod count analysis...")

    # max instances, up rate, down rate
    for dataset in SCALE_DATASETS:
        logging.info(f"Processing scaling stats for: {dataset}")
        df = read_data(dataset, instances_transformer(dataset))
        compute_instance_metrics_cdf(df, dataset)

    logging.info("Processing Instance Lifespan...")
    for dataset in LIFESPAN_DATASETS:
        logging.info(f"Processing lifespan for: {dataset}")
        df = read_data(dataset, instances_transformer(dataset))
        compute_lifespan_cdf(df, dataset)

    # concurrency limit, min scale
    logging.info("Processing Config Limits...")
    for dataset in CONFIG_DATASETS:
        col_name = "concurrency_limit" if "concurrency" in dataset else "min_scale"
        df_conf = read_data(dataset, instances_transformer(dataset))
        compute_config_cdf(df_conf, dataset, col_name)


def generate_stats():
    logging.info("Generating statistics for instances...")
    stats_max_instances = {}
    stats_scale_up = {}
    stats_scale_down = {}

    for dataset in SCALE_DATASETS:
        df_max = pd.read_csv(f"results/instances/csv/{dataset}_max_instances_cdf.csv")
        df_up = pd.read_csv(f"results/instances/csv/{dataset}_scale_up_rate_cdf.csv")
        df_down = pd.read_csv(
            f"results/instances/csv/{dataset}_scale_down_rate_cdf.csv"
        )

        stats_max_instances[dataset] = df_max["max_instances"].to_numpy()
        stats_scale_up[dataset] = df_up["scale_up_rate"].to_numpy()
        stats_scale_down[dataset] = df_down["scale_down_rate"].to_numpy()

    calc_stats(stats_max_instances, "results/instances/stats/max_instances.csv")
    calc_stats(stats_scale_up, "results/instances/stats/scale_up_rate.csv")
    calc_stats(stats_scale_down, "results/instances/stats/scale_down_rate.csv")

    stats_lifespan = {}
    for dataset in LIFESPAN_DATASETS:
        df_life = pd.read_csv(f"results/instances/csv/{dataset}_lifespan_cdf.csv")
        stats_lifespan[dataset] = df_life["lifespan"].to_numpy()
    calc_stats(stats_lifespan, "results/instances/stats/lifespan.csv")

    stats_config = {}
    for dataset in CONFIG_DATASETS:
        col = "concurrency_limit" if "concurrency" in dataset else "min_scale"
        df_cfg = pd.read_csv(f"results/instances/csv/{dataset}_cdf.csv")
        stats_config[dataset] = df_cfg[col].to_numpy()
    calc_stats(stats_config, "results/instances/stats/config_limits.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    plot_metric_cdf(SCALE_DATASETS, "max_instances", "Peak Concurrent Instances")
    plot_metric_cdf(
        SCALE_DATASETS, "scale_up_rate", "Max Scale-Up Rate (Instances/min)"
    )
    plot_metric_cdf(
        SCALE_DATASETS, "scale_down_rate", "Max Scale-Down Rate (Instances/min)"
    )
    plot_lifespan_cdf(LIFESPAN_DATASETS)
    plot_concurrency_limit_cdf("ibm26_concurrency_limit")
    plot_min_scale_cdf("ibm26_min_scale")
