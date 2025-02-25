# src/download_data.py
import os
import subprocess
import tarfile
import zipfile
import requests  # For checking if online
import shutil


def download_file(url, destination_folder):
    """Downloads a file using wget, handling errors."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    filename = url.split('/')[-1]
    filepath = os.path.join(destination_folder, filename)

    # Check if the file already exists.
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return filepath

    try:
        print(f"Downloading {url} to {filepath}...")
        # Use subprocess to call wget.  This is more robust for large files.
        subprocess.run(['wget', url, '-O', filepath], check=True)
        print("Download complete.")
        return filepath
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def extract_file(filepath, destination_folder):
    """Extracts a tar.gz or zip file."""
    try:
        if filepath.endswith('.tar.gz'):
            print(f"Extracting {filepath}...")
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=destination_folder)
            print("Extraction complete.")
        elif filepath.endswith('.zip'):
            print(f"Extracting {filepath}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
            print("Extraction complete.")
        else:
            print(f"Unsupported file type: {filepath}")
            return False
        return True

    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        return False


def move_file(source, destination):
    """Moves the files to the destination"""
    try:
        if not os.path.exists(destination):
            shutil.move(source, destination)
            print(f"Moved {source} to {destination}")
        else:
            print(f"{destination} already exists")
    except FileNotFoundError:
        print(f"{source} does not exist")
    except Exception as e:
        print(f"Error: {e}")

def is_connected():
    """Check if there is an internet connection."""
    try:
        # Try to make a request to a well-known website
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False

def main():
    """Downloads and organizes the HDFS datasets."""

    if not is_connected():
        print("No internet connection.  Cannot download data.")
        return

    # --- HDFS_v1 ---
    hdfs_v1_url = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
    hdfs_v1_archive = download_file(hdfs_v1_url, "data/raw/HDFS_v1")
    if hdfs_v1_archive:
        extract_file(hdfs_v1_archive, "data/raw/HDFS_v1") #extract to raw
        # Move files to the correct processed directory
        move_file("data/raw/HDFS_v1/anomaly_label.csv", "data/processed/HDFS_v1/anomaly_label.csv")
        move_file("data/raw/HDFS_v1/HDFS.npz", "data/processed/HDFS_v1/HDFS.npz")
        move_file("data/raw/HDFS_v1/Event_occurrence_matrix.csv", "data/processed/HDFS_v1/Event_occurrence_matrix.csv")
        move_file("data/raw/HDFS_v1/Event_traces.csv", "data/processed/HDFS_v1/Event_traces.csv")
        move_file("data/raw/HDFS_v1/HDFS.log_templates.csv", "data/processed/HDFS_v1/HDFS.log_templates.csv")

    # --- HDFS_v3 (TraceBench) ---
    hdfs_v3_url = "https://zenodo.org/records/8196385/files/HDFS_v3_TraceBench.zip?download=1"
    hdfs_v3_archive = download_file(hdfs_v3_url, "data/raw/HDFS_v3")
    if hdfs_v3_archive:
        extract_file(hdfs_v3_archive, "data/raw/HDFS_v3")
        # Move files to the correct processed directory
        move_file("data/raw/HDFS_v3/HDFS_v3_TraceBench/HDFS_v3.log_structured.csv", "data/processed/HDFS_v3/HDFS_v3.log_structured.csv")
        move_file("data/raw/HDFS_v3/HDFS_v3_TraceBench/rowNumberResult.csv", "data/processed/HDFS_v3/rowNumberResult.csv")
        move_file("data/raw/HDFS_v3/HDFS_v3_TraceBench/data_process.py", "data/processed/HDFS_v3/data_process.py") #Keep data_process.py
        move_file("data/raw/HDFS_v3/HDFS_v3_TraceBench/eventId.json", "data/processed/HDFS_v3/eventId.json")
        move_file("data/raw/HDFS_v3/HDFS_v3_TraceBench/failure_taskId.json", "data/processed/HDFS_v3/failure_taskId.json")
        move_file("data/raw/HDFS_v3/HDFS_v3_TraceBench/failure_trace.csv", "data/processed/HDFS_v3/failure_trace.csv")
        move_file("data/raw/HDFS_v3/HDFS_v3_TraceBench/normal_taskId.json", "data/processed/HDFS_v3/normal_taskId.json")
        move_file("data/raw/HDFS_v3/HDFS_v3_TraceBench/normal_trace.csv", "data/processed/HDFS_v3/normal_trace.csv")

     # --- Tracebench (Optional, for validation) ---
    tracebench_url = "https://zenodo.org/records/8196385/files/HDFS_v3_TraceBench.zip?download=1"
    tracebench_archive = download_file(tracebench_url, "data/raw/tracebench")

    if tracebench_archive:
        extract_file(tracebench_archive, "data/raw/tracebench") # Extract to raw

if __name__ == "__main__":
    main()