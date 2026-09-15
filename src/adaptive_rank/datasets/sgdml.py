import hashlib
import logging
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import scipy as sp
from sklearn.model_selection import train_test_split

from ..datasets import LabeledDataset, download_file, standardize_dataset

type Array = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class SGDMLDatasetError(Exception):
    "Custom exception type for errors related to the sGDML dataset handling."


class MolecularDynamicsData(NamedTuple):
    "Relevant data from a molecular dynamics simulation, as provided by the sGDML datasets."

    E: Array
    "The energies of the molecular system."

    R: Array
    "The positions of the atoms in the molecular system."


# List of supported sGDML datasets (each identified by an unique key).
SGDML_DATASETS: dict[str, str] = {
    "aspirin": "md17_aspirin",
    "benzene": "md17_benzene2017",
    "ethanol": "md17_ethanol",
    "malonaldehyde": "md17_malonaldehyde",
    "toluene": "md17_toluene",
    "uracil": "md17_uracil",
}


def download_sgdml_dataset(identifier: str, target_path: Path) -> None:
    logger.debug(f"Downloading {identifier} dataset from sGDML")

    # Check if a file with the same name already exists
    if target_path.exists():
        logger.debug("File already exists, not downloading it again.")
        return

    # URL for the file (as provided on the sGDML website)
    file_url = f"https://sgdml.org/secure_proxy.php?file=data/npz/{identifier}.npz"

    logger.debug("Beginning data file download...")
    download_file(file_url, target_path.resolve())
    logger.debug("File downloaded successfully!")


def check_sgdml_dataset_hash(data: Any) -> bool:
    """Checks the MD5 hash of the given sGDML dataset against the expected hash,
    stored in the file itself.

    Can be used to check for incomplete/corrupted downloads,
    but doesn't actually protect against tampering.
    """
    expected_md5 = data["md5"].item()
    assert isinstance(expected_md5, bytes), "Expected MD5 hash must be a bytes object"

    md5_hash = hashlib.md5()

    keys = ["z", "R", "E", "F"]
    for key in keys:
        value = data[key]
        if type(value) is np.ndarray:
            value = value.ravel()

        md5_hash.update(hashlib.md5(value).digest())

    computed_md5 = md5_hash.hexdigest().encode("utf-8")

    if expected_md5 != computed_md5:
        logger.warning(
            f"Dataset hash check failed! {computed_md5.decode('utf-8')} != {expected_md5.decode('utf-8')}"
        )
        return False

    logger.info("Dataset hash check passed")
    return True


def transform_molecular_dynamics_data(positions: Array) -> Array:
    """Convert from a positional representation (each entry represents the 3D positions of N atoms)
    to a pairwise distance representation (each entry represents the pairwise distances between the N atoms).

    This has the advantage of being invariant under isometries (translations, rotations), which is what we want, since only the relative positions of the atoms matter.
    """
    trajectory_length = positions.shape[0]
    num_atoms = positions.shape[1]
    num_distances = num_atoms * (num_atoms - 1) // 2

    pairwise_distances = np.empty(
        shape=(trajectory_length, num_distances), dtype=np.float64
    )
    # Compute pairwise distances between the atoms
    for i in range(len(positions)):
        # After some benchmarking, this seems to be the fastest solution
        pairwise_distances[i, :] = sp.spatial.distance.pdist(
            positions[i], metric="euclidean"
        )

    return pairwise_distances


def process_chemistry_dataset(
    energies: Array,
    positions: Array,
    train_size: float | None = None,
    test_size: float | None = None,
) -> LabeledDataset:
    """Processes a molecular dynamics dataset (as provided by the sGDML datasets)
    into a format suitable for machine learning.
    """
    pairwise_distances = transform_molecular_dynamics_data(positions)

    if test_size is None or test_size > 0:
        # Split the dataset into a training set and a validation set
        X_train, X_test, y_train, y_test = train_test_split(
            pairwise_distances,
            energies,
            train_size=train_size,
            test_size=test_size,
            random_state=7,
        )
    else:
        # Use the entire dataset as the training set, and no validation set
        X_train = pairwise_distances
        y_train = energies
        X_test = None
        y_test = None

    dataset = LabeledDataset(
        cast(np.ndarray, X_train),
        cast(np.ndarray, y_train),
        cast(np.ndarray | None, X_test),
        cast(np.ndarray | None, y_test),
    )

    # Standardize the features (pairwise distances) and the targets (energies)
    return standardize_dataset(dataset)


def load_raw_sgdml_dataset(
    dataset_identifier: str,
    data_directory: Path | None = None,
    download_if_missing: bool = True,
    check_hash: bool = False,
) -> MolecularDynamicsData:
    """Loads an sGDML molecular dynamics dataset, trying first to load it from disk,
    and downloading it if necessary.
    """
    if dataset_identifier not in SGDML_DATASETS:
        raise ValueError(
            f"Dataset identifier must be one of {list(SGDML_DATASETS.keys())}"
        )

    if data_directory is None:
        data_directory = Path("data/sgdml/")

    data_directory.mkdir(parents=True, exist_ok=True)

    dataset_file = data_directory / f"{dataset_identifier}.npz"

    if dataset_file.exists():
        logger.debug(
            f"Dataset file '{dataset_file.name}' already exists, not downloading it again..."
        )
    elif download_if_missing:
        logger.info(
            f"Dataset file '{dataset_file.name}' doesn't exist, downloading it..."
        )
        download_sgdml_dataset(SGDML_DATASETS[dataset_identifier], dataset_file)
    else:
        raise SGDMLDatasetError(
            "Requested sGDML dataset is missing from disk and downloading it has not been requested"
        )

    # Check if the file is very small; it might be that it's actually a 404 error
    if dataset_file.stat().st_size < 512:
        with open(dataset_file, "r", errors="ignore") as fin:
            contents = fin.read()
            if "404" in contents:
                raise SGDMLDatasetError(
                    "Dataset file has been downloaded, but it's actually a 404 error response from the server. Is the dataset identifier correct?"
                )

    # Now that the dataset has been downloaded, we can open it for reading.
    logger.debug("Loading sGDML dataset...")
    data = np.load(dataset_file)

    # print("Available keys in the dataset NPZ file:", data.files)

    if check_hash:
        logger.info("Checking dataset hash...")
        if check_sgdml_dataset_hash(data):
            logger.info("Match: OK")
        else:
            logger.info("Match: FAIL")
            raise RuntimeError("Dataset hash check failed")

    return MolecularDynamicsData(data["E"], data["R"])


def load_and_process_sgdml_dataset(
    dataset_identifier: str,
    num_vectors: int | None = None,
    train_size: float | None = None,
    test_size: float | None = None,
    ignore_cache: bool = False,
    cache_directory: Path | None = None,
    data_directory: Path | None = None,
    check_hash: bool = False,
) -> LabeledDataset:
    """Loads and processes an sGDML molecular dynamics dataset.

    Returns a dataset ready to be used for regression,
    with the features being the pairwise distances between atoms and
    the targets being the energies of the molecular system.
    """
    if cache_directory is None:
        # Use the default cache directory
        cache_directory = Path("cache")

    if not ignore_cache:
        cache_directory.mkdir(parents=True, exist_ok=True)

        N = num_vectors if num_vectors is not None else "all"
        cache_file_path = (
            cache_directory
            / f"sgdml_{dataset_identifier}_N_{N}_train_{train_size}_test_{test_size}.npz"
        )
        if cache_file_path.exists():
            logger.info(
                f"Loading processed dataset from cache file '{cache_file_path}'..."
            )
            data = np.load(cache_file_path)
            return LabeledDataset(
                data["X_train"], data["y_train"], data["X_test"], data["y_test"]
            )

    logger.info("Loading raw sGDML dataset...")

    energies, positions = load_raw_sgdml_dataset(
        dataset_identifier, data_directory, check_hash=check_hash
    )

    # 3 spatial dimensions for each atom
    assert positions.shape[2] == 3

    cut_off = num_vectors
    # Negative cut-off or None = use all vectors
    if cut_off is not None and cut_off > 0:
        available_vectors = len(energies)
        if cut_off > available_vectors:
            raise ValueError(
                f"Requested {cut_off} vectors, but the dataset only contains {available_vectors} vectors"
            )

        # print(f"Limiting data set to first {cut_off} vectors")
        energies = energies[:cut_off]
        positions = positions[:cut_off]

    dataset = process_chemistry_dataset(
        energies, positions, train_size=train_size, test_size=test_size
    )

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
