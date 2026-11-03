import os
import glob
import logging

import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from src.util import savefig_safe, to_csv_safe, get_dataset_style
from src.local import data_path

plt.style.use("src/scientific.mplstyle")


def _latex_safe_label(label: str) -> str:
    # Escape only characters that break TeX parsing in legend labels.
    replacements = {
        "\\": r"\\textbackslash{}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = label
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def compute_cdf(dataset: str, df: dd.DataFrame):
    if "count" in df.columns:
        df_pd = df.groupby("size")["count"].sum().reset_index().compute()
    else:
        df_pd = df.compute()
        df_pd["count"] = 1

    df_sorted = df_pd.sort_values("size").reset_index(drop=True)

    if "latency" not in dataset:
        df_sorted["size"] = df_sorted["size"] / 1024

    total_count = df_sorted["count"].sum()
    df_sorted["cdf"] = df_sorted["count"].cumsum() / total_count

    output_path = os.path.join("results/io/csv", f"{dataset}_cdf.csv")
    to_csv_safe(df_sorted[["size", "cdf"]], output_path)


def plot_request_body_size_cdf(datasets: list[str]) -> None:
    size_datasets = [d for d in datasets if "latency" not in d]

    plt.figure()
    for dataset in size_datasets:
        df = pd.read_csv(f"results/io/csv/{dataset}_cdf.csv")
        label = dataset.replace("_request_body_size", "").title().replace("Ibm", "IBM")
        style = get_dataset_style(dataset)
        plt.step(df["size"], df["cdf"], where="post", label=label, **style)

    plt.xscale("symlog", linthresh=1)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${int(x)}$"))
    plt.xticks([0, 1, 10, 100, 1000, 10000])
    plt.xlim(0, 10000)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.ylim(0, 1)

    plt.xlabel("Request Body Size (KB)")
    plt.ylabel("Requests CDF")
    plt.legend(loc="lower right", handlelength=1.5)
    savefig_safe("results/io/request_body_size_cdf.pdf")


def plot_io_latency_cdf(datasets: list[str]) -> None:
    io_latency_datasets = [d for d in datasets if "latency" in d]

    plt.figure()
    for dataset in io_latency_datasets:
        df = pd.read_csv(f"results/io/csv/{dataset}_cdf.csv")
        labels = {
            "huawei25_read_body_latency": "Read Body",
            "huawei25_write_rsp_latency": "Write Response",
        }
        plt.step(df["size"], df["cdf"], where="post", label=labels[dataset])

    plt.xscale("log")
    plt.xticks([0.000001, 0.0001, 0.01, 1])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.ylim(0, 1)

    def custom_format(x, pos):
        if x >= 1:
            return f"${int(x)}$"
        return f"$10^{{{int(np.log10(x))}}}$"

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(custom_format))

    plt.xlabel("I/O Latency (s)")
    plt.ylabel("Requests CDF")
    plt.legend(loc="lower right", handlelength=1.5)
    savefig_safe("results/io/io_latency_cdf.pdf")


def _build_huawei25_func_id(pdf: pd.DataFrame) -> pd.DataFrame:
    func_name = pd.to_numeric(pdf["funcName"], errors="coerce")
    user_id = pd.to_numeric(pdf["userID"], errors="coerce")
    pool_name = pdf["podID"].astype(str).str.rsplit("-", n=1).str[0]

    out = pd.DataFrame(index=pdf.index)
    func_name_str = func_name.astype("Int64").astype("string")
    user_id_str = user_id.astype("Int64").astype("string")
    pool_name_str = pool_name.astype("string").str.strip()

    out["funcID"] = (
        func_name_str
        + "---"
        + user_id_str
        + "---"
        + pool_name_str
    )
    out["funcNameKey"] = func_name_str
    out["readBodyCost"] = pd.to_numeric(pdf["readBodyCost"], errors="coerce")
    return out[["funcID", "funcNameKey", "readBodyCost"]]


def _expand_weighted_values(
    values: np.ndarray,
    weights: np.ndarray,
    max_points: int = 20000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.array([], dtype=float)

    values = values[valid].astype(float)
    weights = weights[valid].astype(float)
    total_weight = weights.sum()
    if total_weight <= 0:
        return np.array([], dtype=float)

    sample_size = max(1, min(int(total_weight), max_points))
    probs = weights / total_weight
    if rng is None:
        rng = np.random.default_rng(42)
    sample_counts = rng.multinomial(sample_size, probs)
    expanded = np.repeat(values, sample_counts)
    return expanded[expanded > 0]


def plot_huawei25_runtime_io_latency_cdf() -> None:
    day30_files = sorted(glob.glob(f"{data_path}/huawei25/day30/R*/part-*.csv"))
    runtime_file = (
        f"{data_path}/huawei25/trigger_types/R2/df_funcID_runtime_triggerType.csv"
    )

    if not day30_files:
        logging.warning("No Huawei25 Day30 files found; skipping runtime I/O latency violin.")
        return
    regions = sorted({os.path.basename(os.path.dirname(f)) for f in day30_files})
    logging.info(f"Using Huawei25 day30 regions: {', '.join(regions)}")
    if not os.path.exists(runtime_file):
        logging.warning(
            "Huawei25 runtime mapping file missing; skipping runtime I/O latency violin."
        )
        return

    df_day30 = dd.read_csv(
        day30_files,
        usecols=["funcName", "userID", "podID", "readBodyCost"],
        blocksize="64MB",
        assume_missing=True,
        encoding="utf-8",
    )

    meta = pd.DataFrame(
        {
            "funcID": pd.Series(dtype="object"),
            "funcNameKey": pd.Series(dtype="object"),
            "readBodyCost": pd.Series(dtype="float64"),
        }
    )
    df_day30 = df_day30.map_partitions(_build_huawei25_func_id, meta=meta)
    df_day30["funcID"] = df_day30["funcID"].astype("string").str.strip()
    df_day30["funcNameKey"] = df_day30["funcNameKey"].astype("string").str.strip()
    df_day30 = df_day30.dropna(subset=["funcID", "funcNameKey", "readBodyCost"])

    df_runtime = dd.read_csv(
        runtime_file,
        usecols=["funcID", "runtime"],
        dtype={"funcID": "string", "runtime": "string"},
        assume_missing=True,
        encoding="utf-8",
    )
    df_runtime["funcID"] = df_runtime["funcID"].astype("string").str.strip()
    df_runtime["runtime"] = df_runtime["runtime"].astype("string").str.strip()
    df_runtime["funcNameKey"] = df_runtime["funcID"].str.split("---").str[0].str.strip()
    df_runtime["runtime"] = df_runtime["runtime"].replace({"C#": ".NET / C#"})
    df_runtime = df_runtime.dropna(subset=["funcID", "runtime"])

    df = dd.merge(
        df_day30[["funcID", "funcNameKey", "readBodyCost"]],
        df_runtime[["funcID", "runtime"]],
        on="funcID",
        how="inner",
    )
    df = df[(df["readBodyCost"] > 0) & (df["runtime"].notnull()) & (df["runtime"] != "")]

    grouped = (
        df.assign(_count=1)
        .groupby(["runtime", "readBodyCost"])["_count"]
        .sum()
        .reset_index()
        .compute()
        .rename(columns={"_count": "count"})
    )

    if grouped.empty:
        logging.warning(
            "No rows matched on strict funcID. Falling back to funcName-based runtime mapping."
        )

        runtime_lookup = (
            df_runtime[["funcNameKey", "runtime"]]
            .dropna(subset=["funcNameKey", "runtime"])
            .compute()
        )
        if runtime_lookup.empty:
            logging.warning("Runtime lookup is empty; cannot build runtime violin plot.")
            return

        runtime_lookup = (
            runtime_lookup.value_counts(["funcNameKey", "runtime"])
            .rename("n")
            .reset_index()
            .sort_values("n", ascending=False)
            .drop_duplicates("funcNameKey")[["funcNameKey", "runtime"]]
        )

        runtime_lookup["funcNameKey"] = runtime_lookup["funcNameKey"].astype(str)
        runtime_lookup["runtime"] = runtime_lookup["runtime"].astype(str)
        df_day30["funcNameKey"] = df_day30["funcNameKey"].astype(str)

        df_fallback = dd.merge(
            df_day30[["funcNameKey", "readBodyCost"]],
            dd.from_pandas(runtime_lookup, npartitions=1),
            on="funcNameKey",
            how="inner",
        )
        df_fallback = df_fallback[
            (df_fallback["readBodyCost"] > 0)
            & (df_fallback["runtime"].notnull())
            & (df_fallback["runtime"] != "")
        ]
        grouped = (
            df_fallback.assign(_count=1)
            .groupby(["runtime", "readBodyCost"])["_count"]
            .sum()
            .reset_index()
            .compute()
            .rename(columns={"_count": "count"})
        )

        if grouped.empty:
            logging.warning("No merged Huawei25 runtime/readBodyCost rows found.")
            return

    grouped = grouped.rename(columns={"readBodyCost": "size"})
    grouped["size"] = pd.to_numeric(grouped["size"], errors="coerce")
    grouped["count"] = pd.to_numeric(grouped["count"], errors="coerce")
    grouped["runtime"] = grouped["runtime"].astype(str).str.strip()
    grouped = grouped.dropna(subset=["size", "count", "runtime"])
    grouped = grouped[
        (grouped["runtime"] != "")
        & (~grouped["runtime"].str.lower().str.contains("unknown", na=False))
        & (grouped["count"] > 0)
    ]

    if grouped.empty:
        logging.warning("No valid runtime distributions to plot for Huawei25 readBodyCost.")
        return

    to_csv_safe(grouped[["runtime", "size", "count"]], "results/io/csv/huawei25_runtime_read_body_latency_violin.csv")

    runtime_order = (
        grouped.groupby("runtime")["count"].sum().sort_values(ascending=False).index
    )

    data_list: list[np.ndarray] = []
    labels: list[str] = []
    use_tex = bool(plt.rcParams.get("text.usetex", False))
    rng = np.random.default_rng(42)
    for runtime in runtime_order:
        runtime_df = grouped[grouped["runtime"] == runtime]
        expanded = _expand_weighted_values(
            runtime_df["size"].to_numpy(),
            runtime_df["count"].to_numpy(),
            max_points=20000,
            rng=rng,
        )
        if expanded.size == 0:
            continue
        data_list.append(expanded)
        label = str(runtime)
        labels.append(_latex_safe_label(label) if use_tex else label)

    if not data_list:
        logging.warning("No violin-ready runtime samples generated for Huawei25 readBodyCost.")
        return

    fig_width = max(10.0, min(20.0, 1.05 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    parts = ax.violinplot(
        data_list,
        showmeans=False,
        showmedians=True,
        showextrema=True,
        widths=0.8,
    )

    colors = plt.cm.get_cmap("viridis", len(data_list))
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors(i))
        body.set_edgecolor("black")
        body.set_alpha(0.6)

    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        part = parts[part_name]
        part.set_edgecolor("black")
        part.set_linewidth(1.2)
        if part_name == "cmedians":
            part.set_color("#e6091c")

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yscale("log")
    min_val = np.min(np.concatenate(data_list))
    ax.set_ylim(bottom=max(min_val * 0.8, 1e-6))
    ax.set_xlabel("Runtime")
    ax.set_ylabel("Read Body Cost (s)")
    plt.tight_layout()
    savefig_safe("results/io/huawei25_runtime_read_body_latency_cdf.pdf")

