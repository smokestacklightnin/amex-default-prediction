import zipfile
import tempfile
import hashlib
from importlib_resources import files
import logging

logging.basicConfig(level=logging.INFO)

_default_output_path = files("amex-default-prediction.data.raw_data")
_default_checksum_path = files("amex-default-prediction.data.raw_data").joinpath(
    "sha512sums.txt"
)


def _hash_file(filename, block_size=65536):
    h = hashlib.sha512()

    with open(filename, "r") as file:
        chunk = 0
        while chunk != "":
            chunk = file.read(block_size)
            h.update(chunk.encode("utf-8"))

    return h.hexdigest()


def from_kaggle(path=None, checksum_path=None):
    if path is None:
        path = _default_output_path
    if checksum_path is None:
        checksum_path = _default_checksum_path

    try:
        checksums = dict()
        with open(checksum_path, "r") as checksum_file:
            for checksum_line in checksum_file:
                ch, fn = checksum_line.split()
                checksums.update({fn: ch})

        for filename, checksum in checksums.items():
            if _hash_file(path.joinpath(filename)) != checksum:
                raise Exception
        logging.info("All raw data files already downloaded")
    except:
        logging.warning(
            "There was something wrong with the raw data files. Downlading and extracting again."
        )


if __name__ == "__main__":
    from_kaggle()
