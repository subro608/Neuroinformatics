"""Download OpenNeuro ds004504 dataset using direct S3 access (no credentials needed)."""
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

DATASET = "ds004504"
VERSION = "1.0.8"
S3_BUCKET = "openneuro.org"
PREFIX = f"{DATASET}/"
TARGET_DIR = "./eeg_data"

def download_dataset():
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # List all objects in the dataset
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX)

    files_to_download = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            size = obj["Size"]
            # Build local path
            rel_path = key[len(PREFIX):]
            local_path = os.path.join(TARGET_DIR, rel_path)

            # Skip if already downloaded with correct size
            if os.path.exists(local_path) and os.path.getsize(local_path) == size:
                continue

            if size > 0:  # skip empty marker objects
                files_to_download.append((key, local_path, size))

    print(f"Found {len(files_to_download)} files to download")

    for i, (key, local_path, size) in enumerate(files_to_download):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        size_mb = size / (1024 * 1024)
        print(f"[{i+1}/{len(files_to_download)}] Downloading {os.path.basename(local_path)} ({size_mb:.1f} MB)")
        try:
            s3.download_file(S3_BUCKET, key, local_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print("Done!")

if __name__ == "__main__":
    download_dataset()
