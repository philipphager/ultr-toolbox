import numpy as np

from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class LTRDataset(Dataset):
    def __init__(
        self,
        query_ids: np.array,
        x: np.array,
        y: np.array,
        offsets: np.array,
        max_length: int,
    ):
        self.query_ids = query_ids
        self.x = x
        self.y = y
        self.offsets = offsets
        self.max_length = max_length

    def __getitem__(self, idx: int):
        # Limit number of documents per query
        start, end = self.offsets[idx], self.offsets[idx + 1]
        end = min(start + self.max_length, end)

        query_id = self.query_ids[idx]
        n = end - start

        # Pad documents to max_length
        x = np.zeros((self.max_length, self.x.shape[1]))
        x[:n] = self.x[start:end]

        y = np.zeros((self.max_length,))
        y[:n] = self.y[start:end]

        mask = np.zeros((self.max_length,), dtype=bool)
        mask[:n] = True

        return query_id, x, y, mask, n

    def __len__(self):
        return len(self.query_ids)


class SVMRankDataset(LTRDataset):
    def __init__(self, path: str, max_length: int):
        query_ids, x, y, offsets = self._parse(path)
        super(SVMRankDataset, self).__init__(query_ids, x, y, offsets, max_length)

    @staticmethod
    def _parse(path: str):
        print(f"Loading dataset: {path}")

        x, y, query_ids = load_svmlight_file(path, query_id=True)
        x = x.toarray()
        offsets = np.hstack(
            [
                [0],
                np.where(query_ids[1:] != query_ids[:-1])[0] + 1,
                [len(query_ids)],
            ]
        )
        query_ids = query_ids[offsets[:-1]]

        return np.array(query_ids), np.array(x), np.array(y), np.array(offsets)
