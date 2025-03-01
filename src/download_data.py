import os
import requests
import shutil


def download_and_save_file(url, destination_path):
    """Downloads a file from a URL and saves it to the specified path."""
    if not os.path.exists(os.path.dirname(destination_path)):
        os.makedirs(os.path.dirname(destination_path))

    if os.path.exists(destination_path):
        print(f"File already exists: {destination_path}")
        return

    try:
        print(f"Downloading {url} to {destination_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        with open(destination_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        print("Download complete.")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")


def main():
    """Downloads and organizes the HDFS datasets."""

    # --- HDFS_v2 ---
    base_url = "https://github.com/logpai/loghub/raw/master/HDFS/"
    files = [
        "HDFS_2k.log_structured.csv",
        "HDFS_2k.log_templates.csv",
        "HDFS_templates.csv"
    ]
    destination_folder = "data/processed/HDFS_v2"

    for file_name in files:
        file_url = base_url + file_name
        destination_path = os.path.join(destination_folder, file_name)
        download_and_save_file(file_url, destination_path)


if __name__ == "__main__":
    main()
