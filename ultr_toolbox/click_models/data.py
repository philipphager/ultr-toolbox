from typing import Union, Iterable, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ClickDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_col: str = "item_ids",
        target_col="clicks",
    ):
        self.x = np.array(df[feature_col].to_list(), dtype=int)
        self.y = np.array(df[target_col].to_list(), dtype=float)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]

    def n_items(self):
        max_item_id = int(np.max(self.x))
        n_items = len(np.unique(self.x))

        if max_item_id > n_items:
            print(f"Found {n_items} items but an item has max id: {max_item_id}")

        return max(max_item_id + 1, n_items)

    def n_ranks(self):
        return self.x.shape[1]


def np_collate(
    batch: Union[np.ndarray, Iterable[np.ndarray]]
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Collate function from the JAX tutorial with PyTorch data loading:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [np_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
