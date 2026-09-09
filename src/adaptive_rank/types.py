import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]

type Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
