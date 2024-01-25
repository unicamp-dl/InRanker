import hashlib
from enum import Enum

import requests
from tqdm.auto import tqdm


class InRankerDatasets(Enum):
    msmarco_url: str = "https://huggingface.co/datasets/unicamp-dl/InRanker-msmarco/resolve/main/InRanker_msmarco.tsv"
    msmarco_md5: str = "3fc9840ea2a8ad17966dec11c48872bf"
    beir_url: str = "https://huggingface.co/datasets/unicamp-dl/InRanker-BEIR/resolve/main/inpars_beir_dataset.jsonl"
    beir_md5: str = "89d914af6d231a32458fcf84a8de61a6"


def download_file_from_huggingface(url: str, destination: str, checksum: str = None):
    """
    Download a file from HuggingFace and show a progress bar.
    Args:
        url (str): URL to download from.
        destination (str): Path to save the file to.
    """
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024
    checksum = hashlib.md5()

    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc="Downloading"
    )
    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                progress_bar.update(len(chunk))
                file.write(chunk)
                checksum.update(chunk)

    progress_bar.close()

    if checksum is not None and checksum.hexdigest() != checksum:
        raise ValueError(
            f"Downloaded file {destination} does not match checksum {checksum}"
        )
