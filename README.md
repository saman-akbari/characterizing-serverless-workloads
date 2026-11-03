# Characterizing Serverless Workloads

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains the code for a large-scale, cross-platform comparative analysis of eight publicly available Function-as-a-Service (FaaS) workload traces from major providers (Alibaba, Azure, Huawei, IBM) and a scientific computing platform (Globus). By characterizing key dimensions such as invocation patterns, latency, cold start behavior, and resource utilization, we provide a comprehensive and up-to-date view of the serverless ecosystem.

## Setup

### System Requirements
* Python 3.10+
* LaTeX distribution: We use LaTeX to render math and text in plots.

Install dependencies:
```bash
pip install -r requirements.txt
```

### Datasets
Download the datasets from the repositories below. By default, the code looks for a `datasets/` folder in the project root. Ensure the files match the required [folder structure](#folder-structure).

*(Optional: If your data is stored elsewhere, you can override the default directories by setting the `RAW_DATA_PATH` environment variable to point to your downloaded raw files and the `DATA_PATH` environment variable to specify where the cleaned, pre-processed output should be saved).*

- **Alibaba21**: [FaaSNet Dataset](https://github.com/ds2-lab/FaaSNet?tab=readme-ov-file#dataset-downloading)
- **Azure19**: [Azure Functions Dataset 2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md)
- **Azure20**: [Azure Functions Blob Dataset 2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsBlobDataset2020.md)
- **Azure21**: [Azure Functions Invocation Trace 2021](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md)
- **Globus24**: [Zenodo Record](https://zenodo.org/records/10044780)
- **Huawei23**: [Data Release 2023](https://github.com/sir-lab/data-release/blob/main/README_data_release_2023.md)
- **Huawei25**: [Data Release 2025](https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md)
- **IBM26**: [IBM Cloud Code Engine Traces](https://github.com/ubc-cirrus-lab/ibm-cloud-code-engine-traces)


## Usage

### Pipeline

```bash
python main.py
```

This entry point handles everything: it verifies the datasets, runs preprocessors to clean the data, executes the analysis for all metrics, calculates statistics, and creates plots. Generated artifacts are saved into the `results/` directory.

### Manual Usage
If you only want to analyze a single metric, you can run the modules individually.

**1. Preprocess Data**
You must first manually run the preprocessor for each dataset you wish to analyze. Processed data will be saved to `datasets/processed/`.

```bash
# Example: Preprocess the Azure21 and IBM26 datasets
python -m src.preprocessors.azure21
python -m src.preprocessors.ibm26
```

**2. Run the Metric Analysis**

```bash
# Example: Run the latency analysis
python -m src.latency.main

# Example: Run the cold starts analysis
python -m src.cold_starts.main
```

### Folder Structure

Ensure your raw data is placed in the `datasets/` directory using the following hierarchy. The pipeline includes a strict dataset verifier that will check this exact structure before running.

```text
datasets
├── alibaba21
│   ├── region_01.csv
│   └── region_02.csv
├── azure19
│   ├── app_memory_percentiles.anon.d{01..12}.csv
│   ├── function_durations_percentiles.anon.d{01..14}.csv
│   └── invocations_per_function_md.anon.d{01..14}.csv
├── azure20
│   └── azurefunctions-accesses-2020.csv
├── azure21
│   └── AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt
├── globus24
│   ├── GlobusComputeDataset-v1-endpoints.csv
│   ├── GlobusComputeDataset-v1-functions.csv
│   └── GlobusComputeDataset-v1-tasks.csv
├── huawei23
│   └── private_dataset
│       ├── {metric}_minute  # (cpu_limit, cpu_usage, function_delay, instances, memory_limit, memory_usage, platform_delay, requests)
│       │   └── day_{000..234}.csv  # Gaps in days, 141 days total
│       └── {metric}_second  # (function_delay, platform_delay, requests)
│           └── day_{000..234}.csv  # Gaps in days, 141 days total
├── huawei25
│   ├── cold_starts
│   │   └── R{1..5}
│   │       └── day_{00..30}.csv
│   ├── day30
│   │   └── R5
│   │       └── part-{00000..00199}*.csv
│   ├── time_series
│   │   ├── R{1..5}
│   │   │   ├── {metric}  # (cpu_usage_avg, memory_usage_avg, num_pods, readBodyCost_avg, requestBodySize_avg, requests, totalCost_avg, totalCost_frontend_avg, writeRspCost_avg)
│   │   │   │   └── day_{00..30}.csv
│   └── trigger_types
│       └── R2
│           └── df_funcID_runtime_triggerType.csv
└── ibm26
    ├── app_configs.pickle
    └── week_{1..10}.pickle
```

## Common Issues

* **Hardware**: Processing these datasets requires a significant amount of memory and storage. You may limit the data to a subset by modifying the dataloaders. From our experience, all datasets except for IBM26 run well on 64 GB of RAM and 100 GB of disk space. For IBM26, we recommend 256 GB RAM.

## License

This project is licensed under the [Apache License 2.0](LICENSE).