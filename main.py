import random
from functools import partial
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from nde.node import NODE
from nde.utils import mlp_init, mlp_forward, plot_vector_field
from nde.data import generate_data, data_split, mass_spring_damper


@jax.value_and_grad
def train_loss(y_pred, y_true):
    return jnp.mean((y_true - y_pred) ** 2.0)


@partial(jax.jit, static_argnums=0)
def train(node, params, opt_state, y):
    y0 = y[0, :, :]
    y_pred = jax.vmap(node.forward, in_axes=(0, None))(y0, params)
    y_pred = jnp.squeeze(y_pred)
    y1 = y[-1, :, :]
    loss, loss_grads = train_loss(y_pred, y1)
    input_grads, parameter_grads = jax.vmap(node.backward, in_axes=(None, 0, 0))(
        params, y_pred, loss_grads
    )
    input_grads = jnp.mean(input_grads, axis=0)
    parameter_grads = jtu.tree_map(lambda x: jnp.mean(x, axis=0), parameter_grads)
    updates, opt_state = optimizer.update(parameter_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


def validation_loss(y_pred, y_true):
    return jnp.mean((y_true - y_pred) ** 2.0)


@partial(jax.jit, static_argnums=0)
def validate(node, params, y):
    y0 = y[0, :, :]
    y_pred = jax.vmap(node.forward, in_axes=(0, None))(y0, params)
    y_pred = jnp.squeeze(y_pred)
    y1 = y[-1, :, :]
    loss = validation_loss(y_pred, y1)
    return loss


if __name__ == "__main__":
    seed = 42069
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    T0 = 0.0
    T1 = 1.0
    h = 0.01

    dynamics_function = mass_spring_damper
    dynamics_args = [1, 1, 1]
    dim = 2
    width = 10.0
    batch_size = 200

    key, subkey = jax.random.split(key)
    hidden_dim = 20
    hidden_layers = 3
    params = mlp_init(dim, dim, hidden_dim, hidden_layers, subkey)

    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    node = NODE(mlp_forward, T0, T1, h)

    epochs = 100
    pbar = tqdm(range(1, epochs + 1))
    key, subkey = jax.random.split(key)
    y = generate_data(
        dynamics_function,
        dynamics_args,
        dim,
        width,
        batch_size,
        subkey,
        T0,
        T1,
        h,
    )
    y_train, y_val = data_split(y)
    for epoch in pbar:
        try:
            loss_train, params, opt_state = train(node, params, opt_state, y_train)
            loss_val = validate(node, params, y_val)

            pbar.set_postfix(
                {"TL": f"{loss_train.item():.4f}", "VL": f"{loss_val.item():.4f}"}
            )
        except KeyboardInterrupt:
            break

    plot_vector_field(lambda x: mlp_forward(x, params))
    # plot_vector_field(lambda x: mass_spring_damper(None, x, dynamics_args))
