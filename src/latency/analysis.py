import os

import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from src.util import (
    compute_boxplot_statistics,
    compute_weighted_boxplot_statistics,
    savefig_safe,
    to_csv_safe,
    get_dataset_style,
)

plt.style.use("src/scientific.mplstyle")

rng = np.random.default_rng(0)


def compute_cdf(dataset: str, df: dd.DataFrame):
    if "count" in df.columns:
        df_pd = df.groupby("duration")["count"].sum().reset_index().compute()
    else:
        df_pd = df.groupby("duration").size().to_frame("count").reset_index().compute()

    df_sorted = df_pd.sort_values("duration").reset_index(drop=True)
    df_sorted["cdf"] = df_sorted["count"].cumsum() / df_sorted["count"].sum()

    output_path = os.path.join("results/latency/csv", f"{dataset}_cdf.csv")
    to_csv_safe(df_sorted[["duration", "cdf"]], output_path)


def compute_boxplot(dataset, df: dd.DataFrame):
    compute_fn = (
        compute_weighted_boxplot_statistics
        if "count" in df.columns
        else compute_boxplot_statistics
    )
    stats = compute_fn(df, "duration")

    boxplot_data = pd.DataFrame(
        {
            "whislo": [stats["whislo"]],
            "q1": [stats["q1"]],
            "med": [stats["med"]],
            "q3": [stats["q3"]],
            "whishi": [stats["whishi"]],
            "mean": [stats["mean"]],
        }
    )

    output_path = os.path.join("results/latency/csv", f"{dataset}_boxplot.csv")
    to_csv_safe(boxplot_data, output_path)


def compute_infra_cdf(dataset: str, df: dd.DataFrame):
    if "count" in df.columns:
        df_pd = df.groupby(["duration", "infra"])["count"].sum().reset_index().compute()
    else:
        df_pd = (
            df.groupby(["duration", "infra"])
            .size()
            .to_frame("count")
            .reset_index()
            .compute()
        )

    categories = df_pd["infra"].unique()

    for cat in categories:
        subset = (
            df_pd[df_pd["infra"] == cat].sort_values("duration").reset_index(drop=True)
        )
        subset["cdf"] = subset["count"].cumsum() / subset["count"].sum()

        cat_safe = str(cat).lower().replace(".", "_")
        output_path = os.path.join(
            "results/latency/csv", f"{dataset}_{cat_safe}_cdf.csv"
        )
        to_csv_safe(subset[["duration", "cdf"]], output_path)


def compute_violin(dataset: str, df: dd.DataFrame):
    if "count" in df.columns:
        df_pd = df.groupby(["duration", "infra"])["count"].sum().reset_index().compute()
    else:
        df_pd = df.compute()
    output_path = os.path.join("results/latency/csv", f"{dataset}_violin.csv")
    to_csv_safe(df_pd, output_path)


def plot_latency_cdf(datasets: list[str]) -> None:
    ylabel = "Requests CDF"
    if any("total" in d for d in datasets):
        metric = "response_time"
    elif any("overhead" in d for d in datasets):
        metric = "system_overhead"
    elif any("func" in d for d in datasets):
        metric = "execution_time_by_function"
        ylabel = "Functions CDF"
    else:
        metric = "execution_time"

    plt.figure(figsize=(11, 4))
    durations_cdf_list = [
        pd.read_csv(f"results/latency/csv/{dataset}_cdf.csv") for dataset in datasets
    ]

    for i, (df, name) in enumerate(zip(durations_cdf_list, datasets)):
        label = (
            name.replace("_total", "")
            .replace("_overhead", "")
            .replace("_func", "")
            .title().replace("Ibm", "IBM")
        )
        style = get_dataset_style(name)
        plt.step(df["duration"], df["cdf"], where="post", label=label, **style)

    plt.xlabel(f"{metric.replace('_', ' ').title()} (ms)")
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.xticks([1, 10, 100, 1000, 10000, 100000, 3600000])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.gca().xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${int(x)}$")
    )
    plt.xlim(1, 100000000)
    plt.ylim(0, 1)
    plt.legend(
        loc="lower right",
        bbox_to_anchor=(1, 0),
        frameon=True,
        facecolor="white",
        framealpha=1,
        edgecolor="white",
        borderpad=0,
    )

    savefig_safe(f"results/latency/{metric}_cdf.pdf")


def plot_latency_boxplot(datasets: list[str]) -> None:
    if any("total" in d for d in datasets):
        metric = "response_time"
    elif any("overhead" in d for d in datasets):
        metric = "system_overhead"
    elif any("func" in d for d in datasets):
        metric = "execution_time_by_function"
    else:
        metric = "execution_time"

    bp_data = []
    for dataset in datasets:
        data = pd.read_csv(f"results/latency/csv/{dataset}_boxplot.csv").to_dict(
            "records"
        )[0]
        data["fliers"] = []
        if data["whislo"] <= 1:
            data["whislo"] = 1
        bp_data.append(data)

    fig, ax = plt.subplots(figsize=(11, 4))
    bplot = ax.bxp(
        bp_data,
        showmeans=True,
        patch_artist=True,
        boxprops=dict(
            linewidth=1.5, edgecolor="black", facecolor="#e0e0e0", clip_on=False
        ),
        medianprops=dict(linewidth=1.5, color="#e6091c", clip_on=False),
        meanprops=dict(
            marker="^",
            markerfacecolor="#26eb47",
            markeredgecolor="black",
            markersize=10,
            clip_on=False,
        ),
        whiskerprops=dict(linewidth=1.5, color="black", clip_on=False),
        capprops=dict(linewidth=1.5, color="black", clip_on=False),
        widths=0.5,
        shownotches=False,
        showcaps=True,
    )
    for patch in bplot["boxes"]:
        patch.set_facecolor("#e0e0e0")

    ax.set_xticklabels(
        [
            d.replace("_total", "")
            .replace("_overhead", "")
            .replace("_func", "")
            .title()
            .replace("Ibm", "IBM")
            for d in datasets
        ],
        fontsize=22
    )
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (ms)")
    plt.yscale("log")

    last_tick = 1000000000 if metric == "response_time" else 100000000
    ticks = [1, 100, 10000, 1000000, last_tick]
    ax.set_yticks(ticks)

    def custom_format(x, pos):
        if x < 100:
            return f"${int(x)}$"
        return f"$10^{{{int(np.log10(x))}}}$"

    ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_format))
    plt.ylim(0, last_tick)

    savefig_safe(f"results/latency/{metric}_boxplot.pdf")


def plot_infra_cdf(dataset: str) -> None:
    csv_dir = "results/latency/csv"
    files = [
        f
        for f in os.listdir(csv_dir)
        if f.startswith(f"{dataset}_") and f.endswith("_cdf.csv")
    ]

    label_map = {"hpc": "HPC", "local": "Local", "cloud": "Cloud"}

    plt.figure(figsize=(11, 4))
    for f in sorted(files):
        df = pd.read_csv(os.path.join(csv_dir, f))
        raw_cat = f.replace(f"{dataset}_", "").replace("_cdf.csv", "")
        label = label_map.get(raw_cat, raw_cat.title())
        plt.step(df["duration"], df["cdf"], where="post", label=label)

    plt.xlabel("Execution Time (ms)")
    plt.ylabel("Requests CDF")
    plt.xscale("log")
    plt.xticks([1, 10, 100, 1000, 10000, 100000, 3600000])
    plt.xlim(1, 100000000)
    plt.gca().xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${int(x)}$")
    )
    plt.ylim(0, 1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.legend(loc="lower right", title="Globus24")

    savefig_safe(f"results/latency/{dataset}_cdf.pdf")


def plot_latency_violin(datasets: list[str]) -> None:
    for dataset in datasets:
        file_path = f"results/latency/csv/{dataset}_violin.csv"
        df = pd.read_csv(file_path)
        runtime_col = "infra"
        if runtime_col not in df.columns:
            continue

        df = df.dropna(subset=[runtime_col, "duration"])

        title_mapping = {
            "Python2": "Python",
            "Python3": "Python",
            "Node.js": "Node.js",
            "Java": "Java",
            "Go1.x": "Go",
            "C#": ".NET (C\\#)",
            "PHP7.3": "PHP",
            "unknown": "Unknown",
            "http": "HTTP",
            "Custom": "Custom Container",
        }
        df[runtime_col] = df[runtime_col].replace(title_mapping)
        grouped = df.groupby(runtime_col)

        data_list = []
        labels = []

        for runtime, subset in sorted(grouped):
            label_str = str(runtime)
            if "unknown" in label_str.lower():
                continue

            vals = subset["duration"].values

            MAX_SAMPLES = 10_000_000
            if "count" in subset.columns:
                counts = subset["count"].values
                total_count = counts.sum()

                if total_count > MAX_SAMPLES:
                    weights = counts / total_count
                    weights /= weights.sum()

                    samples = rng.choice(vals, size=MAX_SAMPLES, p=weights, replace=True)
                    data_list.append(samples)
                else:
                    data_list.append(np.repeat(vals, counts.astype(np.int64)))
            else:
                if len(vals) > MAX_SAMPLES:
                    data_list.append(rng.choice(vals, size=MAX_SAMPLES, replace=False))
                else:
                    data_list.append(vals)

            labels.append(label_str)

        if not data_list:
            continue

        fig, ax = plt.subplots(figsize=(max(11, 1.2 * len(data_list)), 6))

        parts = ax.violinplot(
            data_list, showmeans=False, showmedians=True, showextrema=True, widths=0.8
        )

        num_runtimes = len(data_list)
        colors = plt.cm.get_cmap("viridis", num_runtimes)

        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors(i))
            pc.set_edgecolor("black")
            pc.set_alpha(0.6)

        for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
            vp = parts[partname]
            vp.set_edgecolor("black")
            vp.set_linewidth(1.5)
            if partname == "cmedians":
                vp.set_color("#e6091c")

        ax.set_xlabel("Runtime")
        ax.set_ylabel("Execution Time (ms)")

        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=40, ha="right")

        ax.set_yscale("log")
        ax.set_ylim(bottom=1)

        ax.set_yticks([1, 10, 100, 1000, 10000])
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        savefig_safe(f"results/latency/{dataset}_violin.pdf")
