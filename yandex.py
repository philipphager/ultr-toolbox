from typing import Iterable

import jax
import numpy as np
import optax
import pandas as pd
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from rich.progress import track
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from ultr_toolbox.models.click_models import PositionBasedModel, train_step, eval_step


def np_collate(batch: Iterable[np.array]):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [np_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class ClickDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = df.iloc[idx]
        x = row.doc_ids
        y = row.click.astype(float)
        return x, y


if __name__ == "__main__":
    path = "data/yandex.parquet"
    df = pd.read_parquet(path)

    train_df, val_df = train_test_split(df)
    train = ClickDataset(train_df)
    val = ClickDataset(val_df)
    train_loader = DataLoader(train, batch_size=128, collate_fn=np_collate)
    val_loader = DataLoader(val, batch_size=128, collate_fn=np_collate)

    early_stop = EarlyStopping(min_delta=0.001, patience=1)
    optimizer = optax.adam(learning_rate=0.003)
    model = PositionBasedModel(n_documents=100_000, n_ranks=10)
    # model = CascadeModel(n_documents=100_000)

    key = jax.random.PRNGKey(0)
    x, y = next(iter(train_loader))
    params = model.init(key, x)
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    n_epochs = 10

    for epoch in range(n_epochs):
        for batch in track(train_loader, description=f"Epoch: {epoch} - Train"):
            model_state, loss = train_step(model_state, batch)

        val_metrics = []

        for batch in track(val_loader, f"Epoch: {epoch} - Validation"):
            val_metric = eval_step(model_state, batch)
            val_metrics.append(val_metric)

        val_metric = pd.DataFrame(val_metrics).mean(axis=0).to_dict()
        print(f"Epoch: {epoch} - Validation", val_metric)
        _, early_stop = early_stop.update(val_metric["perplexity"])

        if early_stop.should_stop:
            print("Stopping early")
            break

    y_predict, debug = model_state.apply_fn(model_state.params, x)
    print(debug.examination)
