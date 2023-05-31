import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from ultr_toolbox.models.click_models import CascadeModel, train_step

if __name__ == "__main__":
    x = jnp.array(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
        ]
    )
    y = jnp.array(
        [
            [1, 0, 0, 0.0],
            [0.5, 0.5, 0, 0.0],
        ]
    )

    optimizer = optax.adam(learning_rate=0.1)
    model = CascadeModel(10)

    key = jax.random.PRNGKey(0)
    params = model.init(key, x)
    model_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    for _ in range(2_000):
        model_state, loss = train_step(model_state, (x, y))
        print(loss)

    y_predict, debug = model_state.apply_fn(model_state.params, x)
    print(y_predict.round(2))
    print(debug)
