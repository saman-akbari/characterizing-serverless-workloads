import os

import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd

from src.util import savefig_safe, to_csv_safe

plt.style.use("src/scientific.mplstyle")


def compute_trigger_distribution(df: dd.DataFrame, dataset: str) -> None:
    distribution = df["trigger"].value_counts().compute().reset_index()
    distribution.columns = ["trigger", "count"]

    total = distribution["count"].sum()
    distribution["percentage"] = (distribution["count"] / total) * 100
    distribution = distribution.sort_values("percentage", ascending=False)

    output_path = os.path.join("results/triggers/csv", f"{dataset}.csv")
    to_csv_safe(distribution, output_path)


def plot_trigger_distribution(datasets: list[str]) -> None:
    for dataset in datasets:
        df = pd.read_csv(f"results/triggers/csv/{dataset}.csv")

        plt.figure(figsize=(8, 4))
        bars = plt.bar(df["trigger"], df["percentage"])

        plt.ylabel("Functions (\\%)")
        plt.xlabel("Trigger Type")
        plt.xticks(rotation=40, ha="right")
        plt.ylim(0, 100)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        savefig_safe(f"results/triggers/{dataset}.pdf")
