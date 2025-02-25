# hdfs-knowledge-graph-causal-rca

This repository contains the code and resources for a project on root cause analysis (RCA) in the Hadoop Distributed File System (HDFS) using knowledge graphs and causal inference. The project leverages trace data to build a knowledge graph representing HDFS components, requests, and operations, and then applies the Root Cause Discovery (RCD) algorithm and causal inference techniques to identify the root causes of anomalies.

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/kulaizki/hdfs-knowledge-graph-causal-rca.git
    cd hdfs-knowledge-graph-causal-rca
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download and Prepare Data:**

    Run the provided script to automatically download and set up the necessary data:

    ```bash
    python src/download_data.py
    ```

    This script will:

    *   Download `HDFS_v3.log_structured.csv` to `data/processed/HDFS_v3/`.
    *   Download and extract the original HDFS dataset to obtain `anomaly_label.csv` and place it in `data/processed/HDFS_v1/`.
    *   Download and extract the `tracebench` dataset to `data/raw/tracebench`.
    *   You need to have `wget` installed on your system. On macOS, install it with `brew install wget`. On Linux, use `sudo apt-get install wget` (or your distribution's package manager). On Windows, you can use a package manager like Chocolatey (`choco install wget`) or download it directly.

4. **Set up the virtual environment kernel (Optional but recommended):**

	```bash
	python -m ipykernel install --user --name=.venv
	```
Then, open the Jupyter Notebook and select the `.venv` kernel.