import logging

import pandas as pd

from src.packages.analysis import (
    imports_barchart,
    latency_imports_scatter,
    plot_code_metric_cdf,
    plot_correlation_heatmap,
)
from src.packages.data_loaders import load_functions, load_tasks
from src.packages.transformers import (
    globus24_functions_transform,
    globus24_tasks_transform,
    merge_data,
)
from src.util import log_timing, setup_logging, calc_stats, to_csv_safe

setup_logging()


@log_timing
def main():
    functions_raw = load_functions()
    tasks_raw = load_tasks()
    functions = globus24_functions_transform(functions_raw)

    tasks = globus24_tasks_transform(tasks_raw)
    df = merge_data(functions, tasks)

    to_csv_safe(functions, "results/packages/csv/globus24_functions.csv")
    to_csv_safe(df, "results/packages/csv/globus24_merged.csv")

    logging.info(f"Merged dataset contains {len(df)} records")
    return functions, df


def generate_stats():
    logging.info("Generating statistics for packages...")
    functions = pd.read_csv("results/packages/csv/globus24_functions.csv")
    calc_stats(
        {
            "imports": functions["imports"].to_numpy(),
            "loc": functions["loc"].to_numpy(),
            "complexity": functions["complexity"].to_numpy(),
        },
        "results/packages/stats/code_metrics_stats.csv",
    )

    merged = pd.read_csv("results/packages/csv/globus24_merged.csv")
    calc_stats(
        {
            "latency": merged["latency"].to_numpy()
        },
        "results/packages/stats/latency_stats.csv",
    )


if __name__ == "__main__":
    main()
    generate_stats()

    imports_barchart()
    plot_code_metric_cdf("loc", "Lines of Code (LoC)")
    plot_code_metric_cdf("complexity", "Cyclomatic Complexity")

    latency_imports_scatter()
    plot_correlation_heatmap()
