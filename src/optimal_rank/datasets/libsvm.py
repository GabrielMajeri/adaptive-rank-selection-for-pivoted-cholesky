import logging
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from ..datasets import LabeledDataset, download_file, standardize_dataset
from ..types import Array

logger = logging.getLogger(__name__)


class LibSVMDatasetError(Exception):
    "Custom exception type for errors related to the libsvm dataset handling."


class LibSVMDatasetKind(StrEnum):
    "Enumeration of the different kinds of libsvm datasets."

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


def download_libsvm_dataset(
    dataset_kind: LibSVMDatasetKind, identifier: str, target_path: Path
) -> None:
    logger.debug(f"Downloading {identifier} dataset from the libsvm website")

    # Check if a file with the same name already exists
    if target_path.exists():
        logger.debug("File already exists, not downloading it again.")
        return

    base_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"

    if dataset_kind == LibSVMDatasetKind.REGRESSION:
        file_url = f"{base_url}/regression/{identifier}"
    elif dataset_kind == LibSVMDatasetKind.CLASSIFICATION:
        file_url = f"{base_url}/classification/{identifier}"
    else:
        raise ValueError("Unknown dataset kind")

    logger.debug("Beginning data file download...")
    download_file(
        file_url,
        target_path.resolve(),
        # Disable TLS verification, since their website's certificate
        # seems to be missing the required Subject Key Identifier field
        verify=False,
    )
    logger.debug("File downloaded successfully!")


def load_raw_libsvm_dataset(
    dataset_kind: LibSVMDatasetKind,
    dataset_identifier: str,
    data_directory: Path | None = None,
    download_if_missing: bool = True,
) -> tuple[Array, Array]:
    "Load a libsvm dataset from the official repository, downloading it if necessary."

    if data_directory is None:
        data_directory = Path("data/libsvm/")

    data_directory.mkdir(parents=True, exist_ok=True)

    dataset_file = data_directory / dataset_identifier

    if dataset_file.exists():
        logger.debug(
            f"Dataset file '{dataset_file.name}' already exists, not downloading it again..."
        )
    elif download_if_missing:
        logger.info(
            f"Dataset file '{dataset_file.name}' doesn't exist, downloading it..."
        )
        download_libsvm_dataset(dataset_kind, dataset_identifier, dataset_file)
    else:
        raise LibSVMDatasetError(
            "Requested libsvm dataset is missing from disk and downloading it has not been requested"
        )

    # Check if the file is very small; it might be that it's actually a 404 error
    if dataset_file.stat().st_size < 512:
        with open(dataset_file, "r", errors="ignore") as fin:
            contents = fin.read()
            if "404" in contents:
                raise LibSVMDatasetError(
                    "Dataset file has been downloaded, but it's actually a 404 error response from the server. Is the dataset identifier correct?"
                )

    # Now that the dataset has been downloaded, we can read it
    logger.debug("Loading LibSVM dataset...")
    data = load_svmlight_file(str(dataset_file))
    assert len(data) == 2, (
        "Expected the dataset to contain exactly two arrays (features and targets)"
    )

    X = data[0].toarray()
    y = data[1]
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return X, y


def load_and_process_libsvm_dataset(
    dataset_kind: LibSVMDatasetKind,
    dataset_identifier: str,
    max_vectors: int | None = None,
    train_size: float | None = None,
    test_size: float | None = None,
    ignore_cache: bool = False,
    cache_directory: Path | None = None,
    data_directory: Path | None = None,
) -> LabeledDataset:
    "Loads and processes a libsvm dataset from the `libsvm` repository."
    if cache_directory is None:
        # Use the default cache directory
        cache_directory = Path("cache")

    if not ignore_cache:
        cache_directory.mkdir(parents=True, exist_ok=True)

        cache_file_path = (
            cache_directory
            / f"libsvm_{dataset_identifier}_N_{str(max_vectors) if max_vectors else 'all'}_train_{train_size}_test_{test_size}.npz"
        )
        if cache_file_path.exists():
            logger.info(
                f"Loading processed dataset from cache file '{cache_file_path}'..."
            )
            data = np.load(cache_file_path)
            return LabeledDataset(
                data["X_train"], data["y_train"], data["X_test"], data["y_test"]
            )

    logger.info("Loading raw libsvm dataset...")

    features, targets = load_raw_libsvm_dataset(
        dataset_kind, dataset_identifier, data_directory
    )

    cut_off = max_vectors
    # Negative cut-off or None = use all vectors
    if cut_off is not None and cut_off > 0:
        features = features[:cut_off]
        targets = targets[:cut_off]

    if test_size is None or test_size > 0:
        # Split the dataset into a training set and a validation set
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            targets,
            train_size=train_size,
            test_size=test_size,
            random_state=7,
        )
    else:
        # Use the entire dataset as the training set, and no validation set
        X_train = features
        y_train = targets
        X_test = None
        y_test = None

    dataset = LabeledDataset(
        cast(np.ndarray, X_train),
        cast(np.ndarray, y_train),
        cast(np.ndarray | None, X_test),
        cast(np.ndarray | None, y_test),
    )

    dataset = standardize_dataset(dataset)

    if not ignore_cache:
        logger.info("Saving processed dataset to cache...")
        np.savez(
            cache_file_path,  # pyright: ignore[reportPossiblyUnboundVariable]
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test or [],
            y_test=dataset.y_test or [],
        )

    return dataset
