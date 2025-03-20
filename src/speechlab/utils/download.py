import hashlib
from pathlib import Path

import requests
from tqdm import tqdm


def calc_file_hash(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_hash_file(file_path: Path) -> str:
    with open(file_path, "r") as f:
        return f.read()


def write_hash_file(file_path: Path, hash: str) -> None:
    with open(file_path, "w") as f:
        f.write(hash)


def download_file(url: str, dest: Path, expected_sha256: str = None) -> None:
    sha_file_name = dest.with_suffix(dest.suffix + ".sha256")
    if dest.exists():
        if expected_sha256:
            if not sha_file_name.exists():
                raise RuntimeError(f"SHA256 file not found for {dest}")

            if read_hash_file(sha_file_name) != expected_sha256:
                raise RuntimeError(f"SHA256 hash verification failed for {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_path = dest.with_suffix(dest.suffix + ".part")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(temp_path, "wb") as f:
            with tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download file '{dest}': {str(e)}")

    file_hash = calc_file_hash(temp_path)
    if expected_sha256 and file_hash != expected_sha256:
        temp_path.unlink()
        raise RuntimeError(f"SHA256 hash verification failed for '{dest}'")

    temp_path.rename(dest)
    write_hash_file(sha_file_name, file_hash)
