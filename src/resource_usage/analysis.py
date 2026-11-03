import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from src.util import savefig_safe, get_dataset_style

plt.style.use("src/scientific.mplstyle")


def plot_memory_cpu_usage(dataset: str) -> None:
    memory_path = f"results/resource_usage/csv/{dataset}_memory_usage.csv"
    cpu_path = f"results/resource_usage/csv/{dataset}_cpu_usage.csv"

    memory = None
    cpu = None

    if os.path.exists(memory_path):
        memory = pd.read_csv(memory_path)

    if os.path.exists(cpu_path):
        cpu = pd.read_csv(cpu_path)

    fig, ax = plt.subplots()

    if memory is not None:
        memory["memory_usage"] = memory["memory_usage_fraction"] * 100
        memory["memory_cdf"] = memory["memory_usage"].rank(method="average", pct=True)
        merged_mem = memory.sort_values("memory_usage")
        ax.plot(
            merged_mem["memory_usage"],
            merged_mem["memory_cdf"],
            label="Memory",
            color="#1f77b4",
        )

    if cpu is not None:
        cpu["cpu_usage"] = cpu["cpu_usage_fraction"] * 100
        cpu["cpu_cdf"] = cpu["cpu_usage"].rank(method="average", pct=True)
        merged_cpu = cpu.sort_values("cpu_usage")
        ax.plot(
            merged_cpu["cpu_usage"], merged_cpu["cpu_cdf"], label="CPU", color="#ff7f0e"
        )

    ax.set_xlabel("Resource Usage (\\%)")
    ax.set_ylabel("Functions CDF")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    savefig_safe(f"results/resource_usage/{dataset}_resource_cdf.pdf")


def plot_memory_cpu_usage_density(dataset: str) -> None:
    memory = pd.read_csv(f"results/resource_usage/csv/{dataset}_memory_usage.csv")
    cpu = pd.read_csv(f"results/resource_usage/csv/{dataset}_cpu_usage.csv")
    merged = pd.merge(memory, cpu, on=["pool"], how="inner")

    # Convert fractions to percentages
    merged["memory_usage"] = merged["memory_usage_fraction"] * 100
    merged["cpu_usage"] = merged["cpu_usage_fraction"] * 100

    plt.style.use("src/scientific.mplstyle")
    fig, ax = plt.subplots()

    base_cmap = plt.get_cmap("plasma")
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    cmap = ListedColormap(colors)

    h = ax.hist2d(
        merged["memory_usage"],
        merged["cpu_usage"],
        bins=50,
        range=[[0, 100], [0, 100]],
        cmap=cmap,
        density=True,
    )

    # show grid
    ax.set_axisbelow(False)

    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label("Density", labelpad=14)

    ax.set_xlabel("Memory Usage (\\%)")
    ax.set_ylabel("CPU Usage (\\%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])

    savefig_safe(f"results/resource_usage/{dataset}_usage_density.pdf")


def plot_memory_limit_cdf(datasets: list[str]) -> None:
    fig, ax = plt.subplots()

    for dataset in datasets:
        df = pd.read_csv(f"results/resource_usage/csv/{dataset}_memory_limit.csv")
        df["cdf"] = df["memory_limit_mb"].rank(method="average", pct=True)
        df = df.sort_values("memory_limit_mb")

        style = get_dataset_style(dataset)
        label = dataset.split("_")[0].capitalize().replace("Ibm", "IBM")
        ax.step(df["memory_limit_mb"], df["cdf"], where="post", label=label, **style)

    memory_ticks = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    ax.set_xscale("log", base=2)
    ax.set_xticks(memory_ticks)

    ax.set_xlim(128, 32768)
    ax.set_xlabel("Memory Limit (MB)")
    ax.set_ylabel("Functions CDF")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(loc="lower right", fontsize=20, bbox_to_anchor=(1.03, -0.08))

    savefig_safe("results/resource_usage/memory_limit_cdf.pdf")


def plot_cpu_limit_cdf(datasets: list[str]) -> None:
    fig, ax = plt.subplots()

    for dataset in datasets:
        path = f"results/resource_usage/csv/{dataset}_cpu_limit.csv"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        df["cdf"] = df["cpu_limit_cores"].rank(method="average", pct=True)
        df = df.sort_values("cpu_limit_cores")

        style = get_dataset_style(dataset)
        label = dataset.split("_")[0].capitalize().replace("Ibm", "IBM")
        ax.step(df["cpu_limit_cores"], df["cdf"], where="post", label=label, **style)

    ax.set_xlabel("CPU Limit (Cores)")
    ax.set_ylabel("Functions CDF")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.legend(loc="lower right", fontsize=20, bbox_to_anchor=(1.05, -0.05), labelspacing=0.08, handletextpad=0.5)

    savefig_safe("results/resource_usage/cpu_limit_cdf.pdf")
