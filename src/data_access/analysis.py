import logging
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.util import to_csv_safe, savefig_safe

plt.style.use("src/scientific.mplstyle")

logging.basicConfig(level=logging.INFO)

FIGSIZE_WIDE: tuple[float, float] = (11, 4)
CDF_YTICKS: list[float] = [0, 0.25, 0.5, 0.75, 1]
BLOB_SIZE_XTICKS: list[int] = [1, 1_000, 1_000_000, 1_000_000_000]
BLOB_SIZE_XLIM: tuple[int, int] = (1, 1_000_000_000)
IAT_XTICKS: list[float] = [0.01, 0.1, 1, 10, 100, 1_000, 1e4, 1e5, 1e6, 1e7]
IAT_XLIM: tuple[float, float] = (0.01, 1e7)
IAT_SCATTER_XTICKS: list[float] = [1, 10, 100, 1_000, 1e4, 1e5, 1e6]
IAT_SCATTER_YTICKS: list[float] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
_prop_colors: list[str] = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
COLOR_READ: str = _prop_colors[0]
COLOR_WRITE: str = _prop_colors[1]
COLOR_REFERENCE_LINE: str = "red"
BAR_ANNOTATION_FONTSIZE: int = 12
SCATTER_ALPHA: float = 0.2


def bytes2human(n, str_format="%(value).2f %(symbol)s"):
    """Convert bytes to human-readable format."""
    symbols = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1000 ** (i + 1)
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return str_format % locals()
    return str_format % dict(symbol=symbols[0], value=n)


def byte_fmt(tick_val, pos):
    """Format bytes for plot tick labels using SI units."""
    if tick_val == 0:
        return r"$0\mathrm{B}$"
    n = float(tick_val)
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    i = 0
    while n >= 1000 and i < len(units) - 1:
        n /= 1000.0
        i += 1

    unit = units[i]

    if unit == "TB" or unit == "PB":
        return r"$%.2f\,\mathrm{%s}$" % (n, unit)
    elif n.is_integer():
        return r"$%d\,\mathrm{%s}$" % (int(n), unit)
    else:
        return r"$%.1f\,\mathrm{%s}$" % (n, unit)


def _format_bytes_si(n_bytes: float) -> str:
    """Human-readable bytes using SI units (KB, MB, GB, TB)."""
    if n_bytes is None or pd.isna(n_bytes):
        return ""
    n = float(n_bytes)
    if n == 0:
        return r"$0\mathrm{B}$"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1000 and i < len(units) - 1:
        n /= 1000.0
        i += 1

    unit = units[i]
    if i == 0:
        return r"$%d\,\mathrm{%s}$" % (int(n), unit)
    elif n >= 100:
        return r"$%.0f\,\mathrm{%s}$" % (n, unit)
    elif n >= 10:
        return r"$%.1f\,\mathrm{%s}$" % (n, unit)
    return r"$%.2f\,\mathrm{%s}$" % (n, unit)


def _annotate_bar_values(
        ax, *, fmt=_format_bytes_si, pad_points=3, min_fraction_of_max=0.0
):
    """Annotate horizontal bars in a seaborn barplot with formatted values."""
    patches = [p for p in ax.patches if p.get_width() > 0]
    if not patches:
        return

    max_w = max(p.get_width() for p in patches)
    threshold = max_w * float(min_fraction_of_max)

    for p in patches:
        w = p.get_width()
        if w <= 0 or w < threshold:
            continue
        y = p.get_y() + p.get_height() / 2.0
        ax.annotate(
            fmt(w),
            (w, y),
            xytext=(pad_points, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=BAR_ANNOTATION_FONTSIZE,
            color="black",
            clip_on=False,
        )


def large_number_fmt(tick_val, pos):
    """Transform large numbers to plot friendly format."""
    if tick_val == 0:
        return r"$0$"
    elif tick_val == 1e-5:
        return r"$10^{-5}$"
    elif tick_val == 1e-4:
        return r"$10^{-4}$"
    elif tick_val == 1e-3:
        return r"$10^{-3}$"
    elif tick_val == 1e-2:
        return r"$0.01$"
    elif tick_val == 0.1:
        return r"$0.1$"
    elif tick_val == 1:
        return r"$1$"
    elif tick_val == 10:
        return r"$10$"
    elif tick_val == 100:
        return r"$100$"
    elif tick_val == 1000:
        return r"$1\,\mathrm{K}$"
    elif tick_val == 10000:
        return r"$10\,\mathrm{K}$"
    elif tick_val == 100000:
        return r"$100\,\mathrm{K}$"
    elif tick_val == 1e6:
        return r"$1\,\mathrm{M}$"
    elif tick_val == 1e7:
        return r"$10\,\mathrm{M}$"
    elif tick_val == 1e8:
        return r"$100\,\mathrm{M}$"
    elif tick_val == 1e9:
        return r"$1\,\mathrm{B}$"
    return r"$%s$" % tick_val


def classify_blob_type(blob_type):
    """
    Groups raw BlobTypes into higher-level categories.
    """
    if pd.isna(blob_type):
        return "Unknown"

    t = str(blob_type).strip().lower()

    # Treat missing/placeholder content-types as unknown.
    if t in {"", "none", "blockblob/", "appendblob/"}:
        return "Unknown"

    # Office / Documents
    if any(x in t for x in ["excel", "sheet", "pdf", "word", "office"]):
        return "Document"

    # Structured Data
    if "json" in t or "xml" in t:
        return "JSON/XML"

    # Media
    if "image" in t:
        return "Image"

    # Archives
    if "zip" in t or "compressed" in t:
        return "Archive"

    # Text / Human Readable
    if any(
            x in t for x in ["text", "plain", "csv", "markdown", "html", "tab-separated"]
    ):
        return "Text"

    # Explicit binary MIME markers.
    if any(x in t for x in ["octet-stream", "x-msdownload", "application/binary"]):
        return "Binary"

    return "Unknown"


def _calculate_iat_stats(df):
    """Calculate IAT statistics for all blobs."""
    df_sorted = df[["AnonBlobName", "time"]].sort_values(["AnonBlobName", "time"])
    df_sorted["iat"] = df_sorted.groupby("AnonBlobName")["time"].diff()
    return (
        df_sorted.groupby("AnonBlobName")["iat"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


# Blob Sizes Analysis -----------------------------------------------------------


def blob_sizes_cdf_to_csv(df: pd.DataFrame, dataset: str) -> None:
    """Generate blob sizes CDF data and save to CSV."""
    start_time = time.time()
    logging.info(f"Calculating blob sizes CDF for dataset: {dataset}")

    # Per access - Read
    read_start = time.time()
    x_read = df[df["Read"] == True]["BlobBytes"].dropna().sort_values().values
    read_data = pd.DataFrame(
        {"BlobBytes": x_read, "type": "Read", "analysis": "per_access"}
    )
    logging.info(
        f"Processed Read access: {len(x_read)} records (elapsed: {time.time() - read_start:.2f}s)"
    )

    # Per access - Write
    write_start = time.time()
    x_write = df[df["Write"] == True]["BlobBytes"].dropna().sort_values().values
    write_data = pd.DataFrame(
        {"BlobBytes": x_write, "type": "Write", "analysis": "per_access"}
    )
    logging.info(
        f"Processed Write access: {len(x_write)} records (elapsed: {time.time() - write_start:.2f}s)"
    )

    # Per access - All
    all_start = time.time()
    x_all = df["BlobBytes"].dropna().sort_values().values
    all_data = pd.DataFrame(
        {"BlobBytes": x_all, "type": "All", "analysis": "per_access"}
    )
    logging.info(
        f"Processed All access: {len(x_all)} records (elapsed: {time.time() - all_start:.2f}s)"
    )

    # Per blob - Read
    blob_read_start = time.time()
    df_blob_read = df[df["Read"] == True][
        ["AnonBlobName", "BlobBytes", "Write"]
    ].drop_duplicates()
    x_blob_read = df_blob_read["BlobBytes"].sort_values().values
    blob_read_data = pd.DataFrame(
        {"BlobBytes": x_blob_read, "type": "Read", "analysis": "per_blob"}
    )
    logging.info(
        f"Processed Read blobs: {len(x_blob_read)} unique blobs (elapsed: {time.time() - blob_read_start:.2f}s)"
    )

    # Per blob - Write
    blob_write_start = time.time()
    df_blob_write = df[df["Write"] == True][
        ["AnonBlobName", "BlobBytes", "Read"]
    ].drop_duplicates()
    x_blob_write = df_blob_write["BlobBytes"].sort_values().values
    blob_write_data = pd.DataFrame(
        {"BlobBytes": x_blob_write, "type": "Write", "analysis": "per_blob"}
    )
    logging.info(
        f"Processed Write blobs: {len(x_blob_write)} unique blobs (elapsed: {time.time() - blob_write_start:.2f}s)"
    )

    # Per blob - All
    blob_all_start = time.time()
    df_blob_all = df.groupby("AnonBlobName")["BlobBytes"].max().reset_index()
    x_blob_all = df_blob_all["BlobBytes"].sort_values().values
    blob_all_data = pd.DataFrame(
        {"BlobBytes": x_blob_all, "type": "All", "analysis": "per_blob"}
    )
    logging.info(
        f"Processed All blobs: {len(x_blob_all)} unique blobs (elapsed: {time.time() - blob_all_start:.2f}s)"
    )

    # Combine all data
    combined = pd.concat(
        [
            read_data,
            write_data,
            all_data,
            blob_read_data,
            blob_write_data,
            blob_all_data,
        ],
        ignore_index=True,
    )

    output_path = os.path.join(
        "results/data_access/blob_sizes_cdf/csv", f"{dataset}.csv"
    )
    to_csv_safe(combined, output_path)

    logging.info(
        f"Blob sizes CDF completed (total elapsed: {time.time() - start_time:.2f}s)"
    )


def blob_sizes_cdf(dataset: str) -> None:
    """Plot blob sizes CDF."""
    start_time = time.time()
    logging.info(f"Plotting blob sizes CDF for dataset: {dataset}")

    input_path = f"results/data_access/blob_sizes_cdf/csv/{dataset}.csv"
    df = pd.read_csv(input_path)

    plt.figure(figsize=FIGSIZE_WIDE)
    ax = plt.gca()
    for access_type in ["Read", "Write", "All"]:
        data = df[(df["type"] == access_type) & (df["analysis"] == "per_access")][
            "BlobBytes"
        ].values
        y = np.arange(len(data)) / len(data)
        ax.plot(data, y, label=access_type)
    ax.set_xscale("log")
    ax.set_ylabel("Blob Access CDF")
    ax.set_xlabel("Blob Size")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(byte_fmt))
    # SI Decimal Ticks
    ax.set_xticks(BLOB_SIZE_XTICKS)
    ax.set_yticks(CDF_YTICKS)
    ax.set_xlim(*BLOB_SIZE_XLIM)
    ax.set_ylim(0, 1)
    savefig_safe(f"results/data_access/blob_sizes_cdf/{dataset}_per_access.pdf")

    plt.figure(figsize=FIGSIZE_WIDE)
    ax = plt.gca()
    for access_type in ["Read", "Write", "All"]:
        data = df[(df["type"] == access_type) & (df["analysis"] == "per_blob")][
            "BlobBytes"
        ].values
        y = np.arange(len(data)) / len(data)
        ax.plot(data, y, label=access_type)
    ax.set_xscale("log")
    ax.set_ylabel("Blob CDF")
    ax.set_xlabel("Blob Size")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(byte_fmt))
    ax.set_xticks(BLOB_SIZE_XTICKS)
    ax.set_yticks(CDF_YTICKS)
    ax.set_xlim(*BLOB_SIZE_XLIM)
    ax.set_ylim(0, 1)
    savefig_safe(f"results/data_access/blob_sizes_cdf/{dataset}_per_blob.pdf")
    logging.info(
        f"Blob sizes CDF plot completed (elapsed: {time.time() - start_time:.2f}s)"
    )


# Invocations per Blob per App Analysis ----------------------------------------


def invocations_per_blob_per_app_cdf_to_csv(df: pd.DataFrame, dataset: str) -> None:
    """Generate invocations per blob per app CDF data."""
    start_time = time.time()
    logging.info(f"Calculating invocations per blob per app for dataset: {dataset}")

    # Count unique blobs and invocations per app
    groupby_start = time.time()
    df_apps_blobs = (
        df.groupby("AnonAppName")
        .agg({"AnonBlobName": "nunique", "AnonFunctionInvocationId": "nunique"})
        .reset_index()
    )
    df_apps_blobs.columns = ["AnonAppName", "AnonBlobName", "AnonFunctionInvocationId"]
    logging.info(f"Groupby completed (elapsed: {time.time() - groupby_start:.2f}s)")

    # Calculate ratio
    invoc_per_blob = (
        (df_apps_blobs["AnonFunctionInvocationId"] / df_apps_blobs["AnonBlobName"])
        .sort_values()
        .values
    )
    invocations = df_apps_blobs["AnonFunctionInvocationId"].sort_values().values
    blobs = df_apps_blobs["AnonBlobName"].sort_values().values

    # Create combined dataframe
    combined = pd.DataFrame(
        {
            "value": np.concatenate([invoc_per_blob, invocations, blobs]),
            "type": (
                    ["Invocations/Blob"] * len(invoc_per_blob)
                    + ["Invocations"] * len(invocations)
                    + ["Blobs"] * len(blobs)
            ),
        }
    )

    output_path = os.path.join(
        "results/data_access/invocations_per_blob_per_app/csv", f"{dataset}.csv"
    )
    to_csv_safe(combined, output_path)

    logging.info(
        f"Invocations per blob per app completed (total elapsed: {time.time() - start_time:.2f}s)"
    )


def invocations_per_blob_per_app_cdf(dataset: str) -> None:
    """Plot invocations per blob per app CDF."""
    start_time = time.time()
    logging.info(f"Plotting invocations per blob per app for dataset: {dataset}")

    input_path = f"results/data_access/invocations_per_blob_per_app/csv/{dataset}.csv"
    df = pd.read_csv(input_path)

    plt.figure(figsize=FIGSIZE_WIDE)

    for metric_type in ["Invocations/Blob", "Invocations", "Blobs"]:
        data = df[df["type"] == metric_type]["value"].values
        y = np.arange(len(data)) / len(data)
        plt.plot(data, y, label=metric_type)

    plt.xlabel("Invocations per Blob")
    plt.ylabel("Apps CDF")
    plt.xscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(large_number_fmt))
    plt.xticks(IAT_XTICKS)
    plt.yticks(CDF_YTICKS)
    plt.legend(loc="lower right")
    plt.xlim(*IAT_XLIM)
    plt.ylim(0, 1)

    savefig_safe(f"results/data_access/invocations_per_blob_per_app/{dataset}.pdf")
    logging.info(
        f"Invocations per blob per app plot completed (elapsed: {time.time() - start_time:.2f}s)"
    )


# Unique Functions per Blob (Data Sharing Scope) ------------------------------


def unique_functions_per_blob_to_csv(df: pd.DataFrame, dataset: str) -> None:
    """Generate shared-blob scope data (>1 unique app) with invocation counts."""
    start_time = time.time()
    logging.info(f"Calculating unique functions per blob for dataset: {dataset}")

    # In this dataset, AnonAppName is used as function/application scope.
    sharing_scope = (
        df.groupby("AnonBlobName", as_index=False)
        .agg(
            UniqueFunctions=("AnonAppName", "nunique"),
            Calls=("AnonBlobName", "size"),
        )
    )

    # Keep only shared blobs, then rank by invocation count.
    sharing_scope = sharing_scope[sharing_scope["UniqueFunctions"] > 1].sort_values(
        ["Calls", "UniqueFunctions"], ascending=[False, False]
    )

    output_path = os.path.join(
        "results/data_access/unique_functions_per_blob/csv", f"{dataset}.csv"
    )
    to_csv_safe(sharing_scope[["AnonBlobName", "UniqueFunctions", "Calls"]], output_path)

    logging.info(
        f"Unique functions per blob completed (total elapsed: {time.time() - start_time:.2f}s)"
    )


def unique_functions_per_blob_cdf(dataset: str) -> None:
    """Plot invocation counts for shared blobs (>1 unique app/function scope)."""
    start_time = time.time()
    logging.info(f"Plotting calls per shared blob for dataset: {dataset}")

    input_path = f"results/data_access/unique_functions_per_blob/csv/{dataset}.csv"
    df = pd.read_csv(input_path)

    plt.figure(figsize=FIGSIZE_WIDE)
    ax = plt.gca()

    if df.empty:
        ax.text(0.5, 0.5, "No shared blobs found", ha="center", va="center")
        ax.set_axis_off()
        savefig_safe(f"results/data_access/unique_functions_per_blob/{dataset}.pdf")
        logging.info(
            f"Unique functions per blob plot completed (elapsed: {time.time() - start_time:.2f}s)"
        )
        return

    rank = np.arange(1, len(df) + 1)
    ax.plot(rank, df["Calls"].values, label="Shared blobs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=1)
    ax.set_xlabel("Shared Blob Rank (1 = most invoked)")
    ax.set_ylabel("Number of Invocations")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(large_number_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(large_number_fmt))
    ax.grid(True, which="both", alpha=0.2)
    plt.legend(loc="lower right")

    savefig_safe(f"results/data_access/unique_functions_per_blob/{dataset}.pdf")
    logging.info(
        f"Calls per shared blob plot completed (elapsed: {time.time() - start_time:.2f}s)"
    )


# Inter-Arrival Time (IAT) Analysis --------------------------------------------


def iat_per_blob_cov_cdf_to_csv(df: pd.DataFrame, dataset: str) -> None:
    """Generate IAT CoV CDF data."""
    start_time = time.time()
    logging.info(f"Calculating IAT CoV for dataset: {dataset}")

    groupby_start = time.time()
    aux_group = _calculate_iat_stats(df)
    logging.info(
        f"IAT aggregation completed (elapsed: {time.time() - groupby_start:.2f}s)"
    )

    # Filter and calculate CoV
    aux_group = aux_group[aux_group["count"] > 1]
    aux_group["CoV"] = aux_group["std"] / aux_group["mean"]
    aux_group = aux_group.sort_values(by="CoV")

    output_path = os.path.join("results/data_access/iat_cov_cdf/csv", f"{dataset}.csv")
    to_csv_safe(aux_group[["CoV"]], output_path)

    logging.info(
        f"IAT CoV CDF completed (total elapsed: {time.time() - start_time:.2f}s)"
    )


def iat_per_blob_cov_cdf(dataset: str) -> None:
    """Plot IAT CoV CDF."""
    start_time = time.time()
    logging.info(f"Plotting IAT CoV CDF for dataset: {dataset}")

    input_path = f"results/data_access/iat_cov_cdf/csv/{dataset}.csv"
    df = pd.read_csv(input_path)

    plt.figure()
    x = df["CoV"].values
    y = np.arange(len(x)) / len(x)
    plt.plot(x, y)
    plt.axvline(1, color=COLOR_REFERENCE_LINE, label="CoV=1")
    plt.xlabel("IAT CoV")
    plt.ylabel("Blobs CDF")
    plt.xlim(0.0001)
    plt.ylim(0, 1)
    plt.yticks(CDF_YTICKS)
    plt.xscale("log")
    plt.legend(loc="upper left")

    savefig_safe(f"results/data_access/iat_cov_cdf/{dataset}.pdf")
    logging.info(
        f"IAT CoV CDF plot completed (elapsed: {time.time() - start_time:.2f}s)"
    )


def iat_per_blob_scatter_to_csv(df: pd.DataFrame, dataset: str) -> None:
    """Generate IAT scatter plot data."""
    start_time = time.time()
    logging.info(f"Calculating IAT scatter data for dataset: {dataset}")

    groupby_start = time.time()
    aux_group = _calculate_iat_stats(df)
    logging.info(
        f"IAT aggregation completed (elapsed: {time.time() - groupby_start:.2f}s)"
    )

    # Filter and calculate CoV
    aux_group = aux_group[aux_group["count"] > 1]
    aux_group["CoV"] = aux_group["std"] / aux_group["mean"]

    output_path = os.path.join("results/data_access/iat_scatter/csv", f"{dataset}.csv")
    to_csv_safe(aux_group[["count", "CoV"]], output_path)

    logging.info(
        f"IAT scatter data completed (total elapsed: {time.time() - start_time:.2f}s)"
    )


def iat_per_blob_scatter(dataset: str) -> None:
    """Plot IAT scatter plot."""
    start_time = time.time()
    logging.info(f"Plotting IAT scatter for dataset: {dataset}")

    input_path = f"results/data_access/iat_scatter/csv/{dataset}.csv"
    df = pd.read_csv(input_path)

    plt.figure()
    x = df["count"].values + 1
    y = df["CoV"].values
    plt.scatter(x=x, y=y, alpha=SCATTER_ALPHA)
    plt.axhline(1, color=COLOR_REFERENCE_LINE, label="CoV=1")
    plt.xlabel("Number of Invocations")
    plt.ylabel("IAT CoV")
    plt.xlim(1)
    plt.ylim(0.00001)
    plt.yscale("log")
    plt.yticks(IAT_SCATTER_YTICKS)
    plt.xscale("log")
    plt.xticks(IAT_SCATTER_XTICKS)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(large_number_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(large_number_fmt))
    plt.legend(loc="lower right")

    savefig_safe(f"results/data_access/iat_scatter/{dataset}.pdf")
    logging.info(
        f"IAT scatter plot completed (elapsed: {time.time() - start_time:.2f}s)"
    )


def bytes_by_blobtype_grouped_to_csv(df: pd.DataFrame, dataset: str) -> None:
    """Generate grouped blob-type transfer data and save to CSV."""
    start_time = time.time()
    logging.info(f"Calculating bytes by blob type (grouped) for dataset: {dataset}")

    plot_df_source = df.copy()
    plot_df_source["BlobCategory"] = plot_df_source["BlobType"].apply(
        classify_blob_type
    )

    # Log the per-row mapping from raw type to grouped category.
    mapping_df = plot_df_source.copy()
    mapping_df["BlobTypePrintable"] = mapping_df["BlobType"].astype("string").fillna("<NA>")
    mapping_counts = (
        mapping_df.groupby(["BlobTypePrintable", "BlobCategory"], as_index=False)
        .size()
        .sort_values(["BlobCategory", "size", "BlobTypePrintable"], ascending=[True, False, True])
    )
    mapping_lines = [
        f"- {row.BlobTypePrintable!r} -> {row.BlobCategory} ({row.size})"
        for row in mapping_counts.itertuples(index=False)
    ]
    logging.info(
        "BlobType -> BlobCategory mapping (rows):\n%s",
        "\n".join(mapping_lines) if mapping_lines else "- <none>",
    )

    df_read = (
        plot_df_source[plot_df_source["Read"] == True]
        .groupby("BlobCategory", as_index=False)["BlobBytes"]
        .sum()
    )
    df_read["Operation"] = "Read"

    df_write = (
        plot_df_source[plot_df_source["Write"] == True]
        .groupby("BlobCategory", as_index=False)["BlobBytes"]
        .sum()
    )
    df_write["Operation"] = "Write"

    plot_df = pd.concat([df_read, df_write], ignore_index=True)

    output_path = os.path.join(
        "results/data_access/bytes_by_blobtype_grouped/csv", f"{dataset}.csv"
    )
    to_csv_safe(plot_df, output_path)

    logging.info(
        f"Bytes by blob type (grouped) completed (elapsed: {time.time() - start_time:.2f}s)"
    )


def bytes_by_blobtype_grouped(dataset: str) -> None:
    """Plot grouped blob-type transfer volume (Read vs Write)."""
    start_time = time.time()
    logging.info(f"Plotting bytes by blob type (grouped) for dataset: {dataset}")

    input_path = f"results/data_access/bytes_by_blobtype_grouped/csv/{dataset}.csv"
    plot_df = pd.read_csv(input_path)

    total_bytes = (
        plot_df.groupby("BlobCategory")["BlobBytes"].sum().sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 4))

    ax = sns.barplot(
        data=plot_df,
        y="BlobCategory",
        x="BlobBytes",
        hue="Operation",
        order=total_bytes.index,
        palette={"Read": COLOR_READ, "Write": COLOR_WRITE},
    )

    plt.xlabel("")
    plt.ylabel("")

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(byte_fmt)
    )
    ax.set_xlim(0, 1.5e12)
    _annotate_bar_values(
        ax, fmt=_format_bytes_si, pad_points=1, min_fraction_of_max=0.0
    )

    plt.legend(title="Operation", loc="lower right")
    plt.tight_layout()
    savefig_safe(f"results/data_access/bytes_by_blobtype_grouped/{dataset}.pdf")

    logging.info(
        f"Bytes by blob type (grouped) plot completed (elapsed: {time.time() - start_time:.2f}s)"
    )
