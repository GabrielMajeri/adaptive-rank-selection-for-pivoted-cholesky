from pathlib import Path
from typing import NamedTuple

import httpx
from sklearn.preprocessing import StandardScaler

from ..types import Array


class LabeledDataset(NamedTuple):
    "A set of data points, ready to be used for a clustering or regression problem."

    X_train: Array
    y_train: Array
    X_test: Array | None
    y_test: Array | None


def download_file(url: str, destination_path: Path, verify: bool = True) -> None:
    """Downloads a file from the given URL and saves it to disk at the provided path.

    Based on https://stackoverflow.com/a/16696317, adapted to use `httpx` instead of `requests`.
    """
    with httpx.stream("GET", url, verify=verify) as r:
        r.raise_for_status()
        with open(destination_path, "wb") as f:
            f.writelines(r.iter_bytes(chunk_size=8192))


def standardize_dataset(dataset: LabeledDataset) -> LabeledDataset:
    """Ensures the dataset's training vectors and targets have zero mean and unit variance,
    then applies the same transform to the testing set.
    """
    features_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train = features_scaler.fit_transform(dataset.X_train)
    if dataset.X_test is not None:
        X_test = features_scaler.transform(dataset.X_test)
    else:
        X_test = None

    y_train = target_scaler.fit_transform(dataset.y_train).ravel()
    if dataset.y_test is not None:
        y_test = target_scaler.transform(dataset.y_test).ravel()
    else:
        y_test = None

    return LabeledDataset(X_train, y_train, X_test, y_test)
