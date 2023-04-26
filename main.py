import jax
import jax.numpy as jnp
import optax
import pandas as pd
import rax
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from rax import ndcg_metric, mrr_metric
from torch.utils.data import DataLoader
from tqdm import tqdm

from ultr_toolbox.cache import cache
from ultr_toolbox.data import SVMRankDataset, numpy_collate
from ultr_toolbox.models import DNN


@jax.jit
def train_step(
    state: TrainState,
    x: jnp.DeviceArray,
    y: jnp.DeviceArray,
    mask: jnp.DeviceArray,
):
    def loss_fn(model, params, x, y, mask):
        y_predict = model.apply_fn(params, x)
        return rax.pairwise_logistic_loss(y_predict, y, where=mask)

    loss, grads = jax.value_and_grad(
        loss_fn,
        argnums=1,  # Position of params in loss_fn
    )(state, state.params, x, y, mask)

    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(
    state: TrainState,
    x: jnp.DeviceArray,
    y: jnp.DeviceArray,
    mask: jnp.DeviceArray,
):
    y_predict = state.apply_fn(state.params, x)

    return {
        "ndcg@5": ndcg_metric(y_predict, y, where=mask, topn=5),
        "ndcg@10": ndcg_metric(y_predict, y, where=mask, topn=10),
        "ndcg": ndcg_metric(y_predict, y, where=mask),
        "mrr": mrr_metric(y_predict, y, where=mask),
    }


def eval(state: TrainState, data_loader: DataLoader):
    val_metrics = []

    for batch in tqdm(data_loader):
        query_id, x, y, mask, n = batch
        val_metrics.append(eval_step(state, x, y, mask))

    return pd.DataFrame(val_metrics).mean(axis=0).to_dict()


@cache("cache/")
def load_data(path, max_length):
    return SVMRankDataset(path, max_length)


def main():
    learning_rate = 0.001
    epochs = 100
    max_length = 100

    train = load_data(
        "/Users/philipphager/.ltr_datasets/dataset/MSLR-WEB10K/Fold1/train.txt",
        max_length,
    )
    val = load_data(
        "/Users/philipphager/.ltr_datasets/dataset/MSLR-WEB10K/Fold1/vali.txt",
        max_length,
    )
    test = load_data(
        "/Users/philipphager/.ltr_datasets/dataset/MSLR-WEB10K/Fold1/test.txt",
        max_length,
    )

    train_loader = DataLoader(train, collate_fn=numpy_collate, batch_size=256)
    val_loader = DataLoader(val, collate_fn=numpy_collate, batch_size=256)
    test_loader = DataLoader(test, collate_fn=numpy_collate, batch_size=256)

    key = jax.random.PRNGKey(0)

    early_stop = EarlyStopping(min_delta=0.001, patience=1)
    optimizer = optax.adam(learning_rate=learning_rate)
    model = DNN([128, 128, 128])

    query_id, x, y, mask, n = train[0]
    params = model.init(key, x)
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    for _ in range(epochs):
        for batch in tqdm(train_loader):
            query_id, x, y, mask, n = batch
            model_state, loss = train_step(model_state, x, y, mask)

        val_metrics = eval(model_state, val_loader)
        _, early_stop = early_stop.update(-val_metrics["ndcg"])
        print("\nVal:", val_metrics)

        if early_stop.should_stop:
            print("Stopping early")
            break

    test_metrics = eval(model_state, test_loader)
    print("Test:", test_metrics)


if __name__ == "__main__":
    main()
