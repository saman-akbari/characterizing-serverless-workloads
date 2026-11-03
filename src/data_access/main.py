import logging
import os
import time

import numpy as np
import pandas as pd

from src.data_access.analysis import (
    blob_sizes_cdf,
    blob_sizes_cdf_to_csv,
    invocations_per_blob_per_app_cdf,
    invocations_per_blob_per_app_cdf_to_csv,
    unique_functions_per_blob_cdf,
    unique_functions_per_blob_to_csv,
    iat_per_blob_cov_cdf,
    iat_per_blob_cov_cdf_to_csv,
    iat_per_blob_scatter,
    iat_per_blob_scatter_to_csv,
    bytes_by_blobtype_grouped,
    bytes_by_blobtype_grouped_to_csv,
)
from src.data_access.data_loaders import read_data
from src.data_access.transformers import data_access_transformer
from src.util import log_timing, setup_logging, calc_stats

setup_logging()

DATASET = "azure20"


@log_timing
def main():
    logging.info("Starting to load and process data...")

    # Load and transform data
    load_start = time.time()
    df = data_access_transformer(DATASET)(read_data(DATASET))
    logging.info(
        f"Dataframe loaded with {len(df)} rows and {len(df.columns)} columns (elapsed: {time.time() - load_start:.2f}s)"
    )

    # Generate blob sizes analysis
    blob_start = time.time()
    blob_sizes_cdf_to_csv(df, DATASET)
    logging.info(
        f"Processed blob sizes CDF for: {DATASET} (elapsed: {time.time() - blob_start:.2f}s)"
    )

    # Generate invocations per blob per app analysis
    invoc_start = time.time()
    invocations_per_blob_per_app_cdf_to_csv(df, DATASET)
    logging.info(
        f"Processed invocations per blob per app for: {DATASET} (elapsed: {time.time() - invoc_start:.2f}s)"
    )

    # Generate shared-blob analysis (sharing scope + calls per shared blob)
    sharing_scope_start = time.time()
    unique_functions_per_blob_to_csv(df, DATASET)
    logging.info(
        f"Processed shared blobs scope/calls for: {DATASET} (elapsed: {time.time() - sharing_scope_start:.2f}s)"
    )

    # Generate inter-arrival time analysis
    iat_cov_start = time.time()
    iat_per_blob_cov_cdf_to_csv(df, DATASET)
    logging.info(
        f"Processed IAT CoV CDF for: {DATASET} (elapsed: {time.time() - iat_cov_start:.2f}s)"
    )

    iat_scatter_start = time.time()
    iat_per_blob_scatter_to_csv(df, DATASET)
    logging.info(
        f"Processed IAT scatter data for: {DATASET} (elapsed: {time.time() - iat_scatter_start:.2f}s)"
    )

    # Generate grouped blob-type transfer analysis
    bytes_by_type_start = time.time()
    bytes_by_blobtype_grouped_to_csv(df, DATASET)
    logging.info(
        f"Processed bytes by blob type (grouped) for: {DATASET} (elapsed: {time.time() - bytes_by_type_start:.2f}s)"
    )


def generate_stats():
    logging.info("Generating statistics for data access...")
    stats_data = {}

    blob_df = pd.read_csv(f"results/data_access/blob_sizes_cdf/csv/{DATASET}.csv")
    for access_type in ["All", "Read", "Write"]:
        for analysis in ["per_access", "per_blob"]:
            stats_data[f"blob_sizes_{access_type.lower()}_{analysis}"] = blob_df[
                (blob_df["type"] == access_type) & (blob_df["analysis"] == analysis)
                ]["BlobBytes"].to_numpy()
    calc_stats(stats_data, f"results/data_access/stats/blob_sizes_{DATASET}.csv")

    invoc_df = pd.read_csv(f"results/data_access/invocations_per_blob_per_app/csv/{DATASET}.csv")
    stats_data_invoc = {
        "invocations_per_blob": invoc_df[invoc_df["type"] == "Invocations/Blob"]["value"].to_numpy(),
        "invocations": invoc_df[invoc_df["type"] == "Invocations"]["value"].to_numpy(),
        "blobs": invoc_df[invoc_df["type"] == "Blobs"]["value"].to_numpy(),
    }
    calc_stats(stats_data_invoc, f"results/data_access/stats/invocations_per_blob_{DATASET}.csv")

    sharing_df = pd.read_csv(f"results/data_access/unique_functions_per_blob/csv/{DATASET}.csv")
    sharing_stats = {}
    if "Calls" in sharing_df.columns:
        sharing_stats["shared_blob_calls"] = sharing_df["Calls"].to_numpy()
    if "UniqueFunctions" in sharing_df.columns:
        sharing_stats["shared_blob_unique_functions"] = sharing_df["UniqueFunctions"].to_numpy()
    if sharing_stats:
        calc_stats(
            sharing_stats,
            f"results/data_access/stats/unique_functions_per_blob_{DATASET}.csv",
        )

    iat_df = pd.read_csv(f"results/data_access/iat_cov_cdf/csv/{DATASET}.csv")
    calc_stats({"iat_cov": iat_df["CoV"].to_numpy()}, f"results/data_access/stats/iat_cov_{DATASET}.csv")

    scatter_df = pd.read_csv(f"results/data_access/iat_scatter/csv/{DATASET}.csv")
    calc_stats({"count": scatter_df["count"].to_numpy()}, f"results/data_access/stats/iat_scatter_{DATASET}.csv")

    bytes_df = pd.read_csv(f"results/data_access/bytes_by_blobtype_grouped/csv/{DATASET}.csv")
    stats_bytes = {
        f"bytes_{row['Operation'].lower()}_{row['BlobCategory'].lower().replace('/', '_').replace(' ', '_')}": np.array(
            [row['BlobBytes']])
        for _, row in bytes_df.iterrows()
    }
    calc_stats(stats_bytes, f"results/data_access/stats/bytes_by_blobtype_{DATASET}.csv")


if __name__ == "__main__":
    main()
    generate_stats()

    logging.info("Plotting blob sizes CDFs...")
    blob_sizes_cdf(DATASET)

    logging.info("Plotting invocations per blob per app CDFs...")
    invocations_per_blob_per_app_cdf(DATASET)

    logging.info("Plotting calls per shared blob (rank plot)...")
    unique_functions_per_blob_cdf(DATASET)

    logging.info("Plotting IAT CoV CDF...")
    iat_per_blob_cov_cdf(DATASET)

    logging.info("Plotting IAT scatter...")
    iat_per_blob_scatter(DATASET)

    logging.info("Plotting bytes by blob type (grouped)...")
    bytes_by_blobtype_grouped(DATASET)
