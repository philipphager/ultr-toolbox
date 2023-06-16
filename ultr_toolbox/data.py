from typing import Iterable, Tuple, Union, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def np_collate(
    batch: Union[np.ndarray, Iterable[np.ndarray]]
) -> Union[np.ndarray, List[np.ndarray]]:
    # This collate function is taken from the JAX tutorial with PyTorch Data Loading
    # https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [np_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class ClickDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.x = np.array(df.doc_ids.tolist(), dtype=int)
        self.y = np.array(df.click.tolist(), dtype=float)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]
