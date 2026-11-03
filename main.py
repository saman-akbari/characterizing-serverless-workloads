import glob
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from src.local import raw_data_path, PROJECT_ROOT
from src.util import setup_logging

setup_logging()


def expand_pattern(pattern: str) -> List[str]:
    """Recursively expands patterns like 'd{01..03}.csv' into ['d01.csv', 'd02.csv', 'd03.csv']."""
    match = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if not match:
        return [pattern]

    start, end = map(int, match.groups())
    width = len(match.group(1))
    prefix, suffix = pattern[: match.start()], pattern[match.end():]

    return [
        res
        for i in range(start, end + 1)
        for res in expand_pattern(f"{prefix}{str(i).zfill(width)}{suffix}")
    ]


def verify_datasets():
    logging.info(f"Verifying Datasets in {raw_data_path}...")

    # Rule types: None (strict file existence) or "count:N" (glob match count)
    rules = [
        ("alibaba21/region_01.csv", None),
        ("alibaba21/region_02.csv", None),
        ("azure19/app_memory_percentiles.anon.d{01..12}.csv", None),
        ("azure19/function_durations_percentiles.anon.d{01..14}.csv", None),
        ("azure19/invocations_per_function_md.anon.d{01..14}.csv", None),
        ("azure20/azurefunctions-accesses-2020.csv", None),
        ("azure21/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt", None),
        ("globus24/GlobusComputeDataset-v1-endpoints.csv", None),
        ("globus24/GlobusComputeDataset-v1-functions.csv", None),
        ("globus24/GlobusComputeDataset-v1-tasks.csv", None),
        ("huawei23/private_dataset/{metric}_minute/day_*.csv", "count:141"),
        ("huawei23/private_dataset/{metric}_second/day_*.csv", "count:141"),
        ("huawei25/cold_starts/R{1..5}/day_{00..30}.csv", None),
        ("huawei25/day30/R5/part-*.csv", "count:200"),
        ("huawei25/time_series/R{1..5}/{metric}/day_{00..30}.csv", None),
        ("huawei25/trigger_types/R2/df_funcID_runtime_triggerType.csv", None),
        ("ibm26/app_configs.pickle", None),
        ("ibm26/week_{1..10}.pickle", None),
    ]

    metrics_huawei23_min = [
        "cpu_limit",
        "cpu_usage",
        "function_delay",
        "instances",
        "memory_limit",
        "memory_usage",
        "platform_delay",
        "requests",
    ]
    metrics_huawei23_sec = ["function_delay", "platform_delay", "requests"]
    metrics_huawei25 = [
        "cpu_usage_avg",
        "memory_usage_avg",
        "num_pods",
        "readBodyCost_avg",
        "requestBodySize_avg",
        "requests",
        "totalCost_avg",
        "totalCost_frontend_avg",
        "writeRspCost_avg",
    ]

    missing = []

    for pat, rule in rules:
        # 1. Expand {metric} placeholders
        patterns = [pat]
        if "{metric}" in pat:
            if "huawei23" in pat and "_minute" in pat:
                metrics = metrics_huawei23_min
            elif "huawei23" in pat and "_second" in pat:
                metrics = metrics_huawei23_sec
            elif "huawei25" in pat:
                metrics = metrics_huawei25
            else:
                metrics = []
            patterns = [pat.replace("{metric}", m) for m in metrics]

        # 2. Verify patterns
        for p in patterns:
            # If pattern uses explicit ranges {1..5}, expand to strict paths and ignore 'count' rule
            if re.search(r"\{\d+\.\.\d+\}", p):
                targets, check_type = expand_pattern(p), "strict"
            else:
                targets, check_type = [p], (
                    "glob" if rule and str(rule).startswith("count:") else "strict"
                )

            for t in targets:
                path = Path(raw_data_path) / t

                if check_type == "glob":
                    expected = int(rule.split(":")[1])
                    found = len(glob.glob(str(path)))
                    if found != expected:
                        missing.append(f"{t} (Found {found}, Expected {expected})")
                else:
                    if not path.exists():
                        missing.append(str(path))

    if missing:
        logging.error(f"Missing {len(missing)} requirements:")
        for m in missing[:20]:
            logging.error(f" - {m}")
        if len(missing) > 20:
            logging.error(f" ... {len(missing) - 20} more")
        sys.exit(1)

    logging.info("Dataset verification successful.")


def run_script(path: Path, desc: str):
    logging.info(f"Running {desc}...")
    start = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    relative_path = path.relative_to(PROJECT_ROOT).with_suffix('')
    module_name = ".".join(relative_path.parts)

    try:
        subprocess.run([sys.executable, "-m", module_name], check=True, env=env)
        logging.info(f"SUCCESS: {desc} ({time.time() - start:.2f}s)")
    except subprocess.CalledProcessError as e:
        msg = f"FAILED: {desc} (Ret: {e.returncode})"
        if e.returncode == -9 or abs(e.returncode) == 9:
            msg += " [Likely OOM]"
        logging.error(msg)


def main():
    logging.info("Starting Pipeline")
    verify_datasets()

    # Preprocessors
    pre_dir = PROJECT_ROOT / "src" / "preprocessors"
    if pre_dir.exists():
        logging.info("--- Preprocessors ---")
        for s in sorted(pre_dir.glob("*.py")):
            if s.name not in ["__init__.py", "main.py"]:
                run_script(s, f"Pre [{s.stem}]")

    # Metrics
    logging.info("--- Metrics ---")
    scripts = [
        p
        for p in PROJECT_ROOT.glob("src/*/main.py")
        if p.parent.name not in ["preprocessors"]
    ]

    # Sort: request_patterns depends on latency_over_time
    get_key = lambda s: (
        0
        if "latency_over_time" in str(s)
        else (2 if "request_patterns" in str(s) else 1)
    )
    for s in sorted(scripts, key=get_key):
        run_script(s, f"Metric [{s.parent.name}]")

    logging.info("Pipeline finished!")


if __name__ == "__main__":
    main()
