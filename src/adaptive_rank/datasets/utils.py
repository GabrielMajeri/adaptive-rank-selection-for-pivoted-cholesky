import numpy as np
from sklearn.model_selection import train_test_split

from ..datasets import LabeledDataset
from .libsvm import LibSVMDatasetKind, load_and_process_libsvm_dataset
from .sgdml import load_and_process_sgdml_dataset


def load_dataset(
    dataset: str,
    num_points: int,
    train_size: float | None = None,
    test_size: float | None = None,
    dimension: int | None = None,
    seed: int | None = None,
) -> LabeledDataset:
    """Loads a labeled regression dataset, either synthetic or from a library,
    and returns it as a `LabeledDataset`.
    """

    if dataset == "random-multivariate-normal":
        if seed is None:
            raise ValueError(
                "Seed must be specified for `random-multivariate-normal` dataset"
            )

        generator = np.random.default_rng(seed)

        if dimension is None:
            raise ValueError(
                "Dimension must be specified for `random-multivariate-normal` dataset"
            )

        X = generator.normal(size=(num_points, dimension))
        y = generator.normal(size=(num_points, 1))

        if train_size is None:
            if test_size is None:
                train_size = 0.8
                test_size = 0.2
            else:
                train_size = 1.0 - test_size

        if test_size is None:
            test_size = 1.0 - train_size

        if train_size + test_size > 1.0:
            raise ValueError("train_size + test_size must be less than or equal to 1.0")

        if train_size <= 0 and test_size <= 0:
            raise ValueError("train_size and test_size cannot both be zero or negative")

        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size, test_size=test_size, random_state=seed
            )
            return LabeledDataset(X_train, y_train, X_test, y_test)

        return LabeledDataset(X, y, None, None)

    elif dataset.startswith("libsvm"):
        dataset_identifier = dataset.split("-", 1)[1]
        labeled_dataset = load_and_process_libsvm_dataset(
            LibSVMDatasetKind.REGRESSION,
            dataset_identifier,
            max_vectors=num_points,
            train_size=train_size,
            test_size=test_size,
        )
        return labeled_dataset

    elif dataset.startswith("sgdml"):
        dataset_identifier = dataset.split("-", 1)[1]
        labeled_dataset = load_and_process_sgdml_dataset(
            dataset_identifier,
            max_vectors=num_points,
            train_size=train_size,
            test_size=test_size,
        )
        return labeled_dataset

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
