import zipfile
import tempfile
import hashlib
from importlib_resources import files
import logging
import kaggle
import pathlib

_competition_name = "amex-default-prediction"
_module_name = "amex_default_prediction"
_default_output_path = files(
    _module_name + ".data.raw_data",
)
_default_checksum_path = files(
    _module_name + ".data.raw_data",
).joinpath("sha512sums.txt")


def _hash_file(filename, block_size=(2**16 - 1)):
    filename = pathlib.Path(filename)
    h = hashlib.sha512()
    logging.debug("Calculating SHA512 hash of " + str(filename))
    with open(filename, "r") as file:
        while (chunk := file.read(block_size)) != "":
            h.update(chunk.encode("utf-8"))

    return h.hexdigest()


def _verify_raw_data(path=None, checksum_path=None):
    if path is None:
        path = _default_output_path
    if checksum_path is None:
        checksum_path = _default_checksum_path
    path, checksum_path = pathlib.Path(path), pathlib.Path(checksum_path)

    checksums = dict()

    with open(checksum_path, "r") as checksum_file:
        for checksum_line in checksum_file:
            ch, fn = checksum_line.split()
            checksums.update({fn: ch})

    for filename, checksum in checksums.items():
        filepath = path.joinpath(filename)
        logging.debug(
            "Verifying " + str(filepath) + " if it exists",
        )

        if not filepath.is_file():
            logging.debug(str(filepath) + " doesn't exist")
            return False
        if _hash_file(filepath) != checksum:
            logging.debug("There was an error with the checksum.")
            return False
        else:
            logging.debug(
                "SHA512 hash of "
                + filename
                + " was successfully calculated and verified.",
            )

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
            tmpdirpath = pathlib.Path(tmpdirname)
            kaggle.api.competition_download_files(
                _competition_name,
                path=tmpdirpath,
                quiet=False,
            )

            with zipfile.ZipFile(
                tmpdirpath.joinpath(_competition_name + ".zip")
            ) as data_zip:
                data_zip.extractall(path=path)
        logging.info("Successfully downloaded and extracted raw data files.")
    else:
        logging.info("All raw data files already downloaded")
