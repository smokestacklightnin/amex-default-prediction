import zipfile
import tempfile
import hashlib
from importlib_resources import files


def _hash_file(filename, block_size=65536):
    h = hashlib.sha512()

    with open(filename, "r") as file:
        chunk = 0
        while chunk != "":
            chunk = file.read(block_size)
            h.update(chunk.encode("utf-8"))

    return h.hexdigest()


# print(_hash_file("raw_data/sample_submission.csv"))

# print(_hash_file("raw_data/test_data.csv"))


def from_kaggle(path):
    try:
        pass
    except:
        pass
