import logging
import os

import dask.dataframe as dd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from src.util import to_csv_safe, savefig_safe, get_dataset_style

plt.style.use("src/scientific.mplstyle")

logging.basicConfig(level=logging.INFO)


def cold_starts_cdf_to_csv(df: dd.DataFrame, dataset: str) -> None:
    logging.info(f"Calculating cdf and saving for dataset: {dataset}")

    df = df.compute()
    df = df.sort_values("cold_start").reset_index(drop=True)

    df["cdf_percent"] = (df.index + 1) / len(df)

    output_path = os.path.join(
        "results/cold_starts/cold_starts_cdf/csv", f"{dataset}.csv"
    )
    to_csv_safe(df[["cold_start", "cdf_percent"]], output_path)


def cold_starts_cdf(datasets: list[str]) -> None:
    logging.info(f"Plotting cold start CDF for datasets: {datasets}")

    plt.figure()
    plt.xscale("symlog", linthresh=10)
    ax = plt.gca()
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000])
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: rf"$10^{{{int(np.log10(x))}}}$" if x > 10 else f"${int(x)}$"
        )
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    max_cold_start = 0
    for dataset in datasets:
        input_path = f"results/cold_starts/cold_starts_cdf/csv/{dataset}.csv"
        df = pd.read_csv(input_path)
        max_cold_start = max(max_cold_start, df["cold_start"].max())

        style = get_dataset_style(dataset)
        plt.plot(
            df["cold_start"], df["cdf_percent"], label=dataset.capitalize().replace("Ibm", "IBM"), **style
        )

    ax.set_xlim(left=0, right=max_cold_start)
    plt.xlabel("Cold Start Latency (s)")
    plt.ylabel("Requests CDF")
    plt.legend(loc="lower right")

    savefig_safe(f"results/cold_starts/cold_starts_cdf/cold_starts_cdf.pdf")


def cold_starts_violin_to_csv(
        ddf: dd.DataFrame,
        dataset: str,
        latency_col: str = "cold_start",
        runtime_col: str = "runtime",
) -> None:
    df = ddf[[runtime_col, latency_col]].dropna()
    final_df = df.compute().reset_index(drop=True)
    output_path = os.path.join(
        "results/cold_starts/runtime_violin/csv", f"{dataset}_violin_data.csv"
    )
    to_csv_safe(final_df, output_path)


def cold_starts_violin(dataset: str, runtime_col: str = "runtime"):
    df = pd.read_csv(
        f"results/cold_starts/runtime_violin/csv/{dataset}_violin_data.csv"
    )

    all_runtimes = sorted(df[runtime_col].unique())

    data_list = []
    labels = []

    for r in all_runtimes:
        label_str = str(r)
        if label_str.lower() in ["unknown", "debian", "http"]:
            continue

        subset = df[df[runtime_col] == r]["cold_start"].values
        if len(subset) > 0:
            data_list.append(subset)
            labels.append(label_str)

    if not data_list:
        return

    fig, ax = plt.subplots(figsize=(11, 5))

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

    ax.set_ylabel("Cold Start Time (s)")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")

    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(bottom=0)

    y_ticks_alibaba21 = [1, 10, 100, 200]
    y_ticks_huawei25 = [1, 10, 100, 500]
    if dataset == "alibaba21":
        ax.set_yticks(y_ticks_alibaba21)
    elif "huawei25" in dataset:
        ax.set_yticks(y_ticks_huawei25)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    savefig_safe(f"results/cold_starts/runtime_violin/{dataset}.pdf")


def cold_starts_breakdown_to_csv(df: dd.DataFrame, dataset: str) -> None:
    logging.info(f"Calculating breakdown and saving for dataset: {dataset}")

    cost_names = [
        "podAllocationCost",
        "deployCodeCost",
        "deployDependencyCost",
        "schedulingCost",
    ]
    df_grouped = df.groupby("day")[cost_names].mean().compute().sort_index()
    df_grouped = df_grouped.reset_index()

    output_path = os.path.join("results/cold_starts/breakdown/csv", f"{dataset}.csv")
    to_csv_safe(df_grouped, output_path)


def cold_starts_breakdown(dataset: str) -> None:
    input_path = f"results/cold_starts/breakdown/csv/{dataset}.csv"
    df = pd.read_csv(input_path)

    cost_names = [
        "podAllocationCost",
        "deployCodeCost",
        "deployDependencyCost",
        "schedulingCost",
    ]
    legend_labels = {
        "podAllocationCost": "Allocation",
        "deployCodeCost": "Deploy Code",
        "deployDependencyCost": "Deploy Dependencies",
        "schedulingCost": "Network, Routing, Scheduling",
    }

    fig, ax = plt.subplots()
    bottoms = np.zeros(len(df))

    df = df.sort_values("day")
    days = df["day"].values

    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(cost_names)))
    for i, cost_name in enumerate(cost_names):
        ax.bar(
            days,
            df[cost_name],
            bottom=bottoms,
            label=legend_labels[cost_name],
            color=colors[i],
            alpha=0.6
        )
        bottoms += df[cost_name].values

    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_xlabel("Day")
    ax.set_ylabel("Cold Start Time (s)")

    create_cold_starts_breakdown_legend()
    savefig_safe(f"results/cold_starts/breakdown/{dataset}.pdf")


def create_cold_starts_breakdown_legend() -> None:
    plt.figure()
    ax = plt.gca()

    cost_names = [
        "podAllocationCost",
        "deployCodeCost",
        "deployDependencyCost",
        "schedulingCost",
    ]
    legend_labels = {
        "podAllocationCost": "Allocation",
        "deployCodeCost": "Deploy Code",
        "deployDependencyCost": "Deploy Dependencies",
        "schedulingCost": "Network, Routing, Scheduling",
    }

    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, len(cost_names)))
    handles = []
    for i, cost_name in enumerate(cost_names):
        handles.append(mpatches.Patch(color=colors[i], alpha=0.6, label=legend_labels[cost_name]))

    ax.axis("off")

    legend = plt.legend(handles=handles, ncol=2, frameon=False, loc="center", handlelength=1.5)
    export_legend(legend, "results/cold_starts/breakdown/legend.pdf")


def cold_starts_rate_to_csv(df: dd.DataFrame, dataset: str) -> None:
    logging.info(f"Calculating rate and saving for dataset: {dataset}")
    df = df.compute() if isinstance(df, dd.DataFrame) else df

    if "day" in df.columns:
        df = df.sort_values("day").reset_index(drop=True)

    output_path = os.path.join("results/cold_starts/rate/csv", f"{dataset}.csv")
    to_csv_safe(df, output_path)


def cold_starts_rate(dataset: str) -> None:
    input_path = f"results/cold_starts/rate/csv/{dataset}.csv"
    df = pd.read_csv(input_path)
    df = df.sort_values("day")

    plt.figure()
    ax = plt.gca()

    rate_col = "rate_req" if "rate_req" in df.columns else "rate"
    has_func = "rate_func" in df.columns

    line_req = plt.plot(df["day"], df[rate_col], marker="o", label="Requests")
    color_req = line_req[0].get_color()

    color_func = None
    if has_func:
        line_func = plt.plot(df["day"], df["rate_func"], marker="o", label="Functions")
        color_func = line_func[0].get_color()

    max_req = df[rate_col].max()
    max_func = df["rate_func"].max() if has_func else 0
    should_annotate = (max_req < 10) or (has_func and max_func < 10)

    if should_annotate:
        huawei_offset = 6
        default_offset = 12
        collision_step = 20

        is_huawei = "huawei25" in dataset
        base_offset = huawei_offset if is_huawei else default_offset

        y_pos_req = max_req + base_offset
        y_pos_func = max_func + base_offset

        if not is_huawei and has_func and abs(y_pos_func - y_pos_req) < 8:
            if y_pos_func >= y_pos_req:
                y_pos_func = y_pos_req + collision_step
            else:
                y_pos_req = y_pos_func + collision_step

        ticks = ax.get_xticks()
        for tick in ticks:
            row = df[df["day"] == tick]
            if row.empty:
                continue

            val_req = row[rate_col].iloc[0]
            ax.text(tick, y_pos_req, f"{val_req:.2f}\\%",
                    ha="center", va="bottom", color=color_req, fontsize=30,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1), clip_on=False)

            if has_func:
                val_func = row["rate_func"].iloc[0]
                ax.text(tick, y_pos_func, f"{val_func:.2f}\\%",
                        ha="center", va="bottom", color=color_func, fontsize=30,
                        bbox=dict(facecolor='white', edgecolor='none', pad=1), clip_on=False)

    plt.xlabel("Day")
    plt.ylabel("Cold Start Rate (\\%)")
    plt.ylim(0, 100)
    plt.yticks([0, 25, 50, 75, 100])

    create_cold_start_legend()
    savefig_safe(f"results/cold_starts/rate/{dataset}.pdf")


def create_cold_start_legend() -> None:
    plt.figure()
    ax = plt.gca()

    ax.plot([], [], marker="o", label="Requests")
    ax.plot([], [], marker="o", label="Functions")

    ax.axis("off")

    legend = plt.legend(ncol=2, frameon=False, loc="center", handlelength=1.5)
    export_legend(legend, "results/cold_starts/rate/legend.pdf")


def export_legend(legend, filename="legend.pdf"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    savefig_safe(filename, bbox_inches=bbox, pad_inches=0)
