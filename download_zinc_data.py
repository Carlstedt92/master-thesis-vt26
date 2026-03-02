import os
path_to_url = "data/zinc/ZINC-downloader-2D-smi.uri"
DOWNLOAD_DIR = "data/zinc/zinc_data"

with open(path_to_url, "r") as f:
    for url in f:
        url = url.strip()
        filename = url.split("/")[-1]
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        print(f"Downloading {filename} from {url}...")
        os.system(f"wget -O {file_path} {url}")