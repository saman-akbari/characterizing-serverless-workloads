import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

from src.util import savefig_safe

plt.style.use("src/scientific.mplstyle")

logging.basicConfig(level=logging.INFO)


def imports_barchart() -> None:
    df = pd.read_csv("results/packages/csv/globus24_functions.csv")
    import_counts = df["imports"].value_counts().sort_index()
    import_counts = import_counts.groupby(lambda x: x if x < 3 else "3+").sum()
    import_perc = import_counts / import_counts.sum() * 100

    labels = [
        str(int(x)) if isinstance(x, (int, float)) else x for x in import_perc.index
    ]

    fig, ax = plt.subplots()
    ax.bar(
        labels,
        import_perc.to_numpy() / 100,
        color="black",
    )

    ax.set_xlabel("Imports")
    ax.set_ylabel("Functions CDF")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    savefig_safe(f"results/packages/globus24_imports_barchart.pdf")


def plot_code_metric_cdf(metric: str, xlabel: str) -> None:
    df = pd.read_csv("results/packages/csv/globus24_functions.csv")
    data = df[metric].sort_values()
    cdf = np.arange(1, len(data) + 1) / len(data)

    fig, ax = plt.subplots()
    ax.step(data, cdf, where="post")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Functions CDF")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=1)
    savefig_safe(f"results/packages/globus24_{metric}_cdf.pdf")


def latency_imports_scatter() -> None:
    df = pd.read_csv("results/packages/csv/globus24_merged.csv")
    df["imports_binned"] = df["imports"].apply(lambda x: x if x < 3 else 3)

    SAMPLE_SIZE = 1000
    if len(df) > SAMPLE_SIZE:
        df_sample = df.sample(SAMPLE_SIZE, random_state=0)
    else:
        df_sample = df

    fig, ax = plt.subplots()

    sns.scatterplot(
        x="imports_binned",
        y="latency",
        data=df_sample,
        color="black",
        alpha=0.3,
        s=10,
        ax=ax,
    )

    means = df.groupby("imports_binned")["latency"].mean().reindex([0, 1, 2, 3])
    if len(means) >= 2:
        x_for_fit = means.index.to_numpy().astype(float)
        y_for_fit = means.to_numpy().astype(float)

        positive_mask = y_for_fit > 0
        x_pos = x_for_fit[positive_mask]
        y_pos = y_for_fit[positive_mask]
        slope_log, intercept_log = np.polyfit(x_pos, np.log(y_pos), 1)

        x_bins = np.array([0, 1, 2, 3], dtype=float)
        log_pred = slope_log * x_bins + intercept_log
        y_pred = np.exp(log_pred)
        mask_plot = ~means.isna().to_numpy().astype(bool)
        ax.plot(x_bins[mask_plot], y_pred[mask_plot], color="red", linewidth=2)

    ax.set_xlabel("Imports")
    ax.set_ylabel("Response Latency (s)")
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100, 1000, 100000])
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([0, 1, 2, "3+"])

    savefig_safe("results/packages/globus24_latency_imports_scatter.pdf")


def plot_correlation_heatmap() -> None:
    df = pd.read_csv("results/packages/csv/globus24_merged.csv")
    corr_df = df[["latency", "imports", "loc", "complexity"]]

    corr_matrix = corr_df.corr(method="spearman")

    new_labels = ["Resp. Time (s)", "Imports", "Lines of Code", "Cyc. Complexity"]
    corr_matrix.columns = new_labels
    corr_matrix.index = new_labels

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1
    )

    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)

    savefig_safe("results/packages/globus24_correlation_heatmap.pdf")
