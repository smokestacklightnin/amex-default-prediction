import zipfile
import tempfile
import hashlib
from importlib_resources import files
import logging
import kaggle
import pathlib

logging.basicConfig(
    level=logging.DEBUG,
)


_competition_name = "amex-default-prediction"
_default_output_path = files(
    _competition_name + ".data.raw_data",
)
_default_checksum_path = files(
    _competition_name + ".data.raw_data",
).joinpath("sha512sums.txt")


def _hash_file(filename, block_size=65536):
    h = hashlib.sha512()

    with open(filename, "r") as file:
        chunk = 0
        while chunk != "":
            chunk = file.read(block_size)
            h.update(chunk.encode("utf-8"))

    return h.hexdigest()


def _verify_raw_data(path=None, checksum_path=None):
    if path is None:
        path = _default_output_path
    if checksum_path is None:
        checksum_path = _default_checksum_path

    checksums = dict()
    with open(checksum_path, "r") as checksum_file:
        for checksum_line in checksum_file:
            ch, fn = checksum_line.split()
            checksums.update({fn: ch})

    for filename, checksum in checksums.items():
        if _hash_file(path.joinpath(filename)) != checksum:
            return False

    return True


def from_kaggle(path=None, checksum_path=None):
    if path is None:
        path = _default_output_path
    if checksum_path is None:
        checksum_path = _default_checksum_path
    path, checksum_path = pathlib.Path(path), pathlib.Path(checksum_path)

    if not _verify_raw_data(path, checksum_path):
        logging.warning(
            "There was something wrong with the raw data files. Downlading and extracting again."
        )
        with tempfile.TemporaryDirectory() as tmpdirname:
            kaggle.api.competition_download_files(
                _competition_name,
                path=tmpdirname,
                quiet=False,
            )
            print(type(tmpdirname), tmpdirname)
            with zipfile.ZipFile(
                pathlib.Path(tmpdirname).joinpath(_competition_name + ".zip")
            ) as data_zip:
                data_zip.extractall(path=path)
        logging.info("Successfully downloaded and extracted raw data files.")
    else:
        logging.info("All raw data files already downloaded")


if __name__ == "__main__":
    from_kaggle()
