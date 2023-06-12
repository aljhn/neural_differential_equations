import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import diffrax


seed = 42069
random.seed(seed)
np.random.seed(seed)
key = jax.random.PRNGKey(seed)


def mass_spring_damper(t, y, args):
    m, d, k = args
    A = jnp.array([[0, 1], [-k / m, -d / m]])
    return y @ A.T


# Generate trajectories by sampling random initial values and integrating those
def generate_data(data_term, data_args, dim, batch_size, key, solver, T0, T1, h, saveat):
    y0 = jax.random.ball(key, d=dim, p=2, shape=(batch_size,)) * 10
    solution = diffrax.diffeqsolve(data_term, solver, t0=T0, t1=T1, dt0=h, y0=y0, saveat=saveat, args=data_args, max_steps=None, adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=None))
    return solution.ys


T0 = 0.0
T1 = 1.0
h = 0.01
saveat = diffrax.SaveAt(ts=jnp.array([T0, T1]))
solver = diffrax.Dopri5()

data_term = diffrax.ODETerm(mass_spring_damper)
data_args = [1, 1, 1]
dim = 2
batch_size = 100


# Define models as a list of nodes per layer
# Return the parameters as a list of dictionaries containing arrays
def model_init(model_def, key):
    subkeys = jax.random.split(key, num=(len(model_def) - 1) * 2)
    params = []
    for i in range(len(model_def) - 1):
        layer = {
            "weights": jax.random.normal(subkeys[i], (model_def[i], model_def[i + 1])),
            "bias": jax.random.normal(subkeys[i + len(model_def) - 1], (model_def[i + 1],))
        }
        params.append(layer)
    return params


key, subkey = jax.random.split(key)
model_def = [2, 50, 50, 2]
params = model_init(model_def, subkey)


optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)


# Use the model definition to extract the individual parameters from the total vector
# The forward pass is computed as a standard fully connected neural network
# The tanh activation function is applied at every layer except the last
def model_forward(x, params):
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return x


model_term = diffrax.ODETerm(lambda t, y, args: model_forward(y, args))
    

# Neural ODE forward pass
# Use the input as an initial value and integrate it using the model as the dynamics.
# Using the parameters max_steps=None and adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=None)
# means that diffrax disables automatic differentiation, as this is implemented manually with the augmented dynamics
def forward(y0, params):
    solution = diffrax.diffeqsolve(model_term, solver, t0=T0, t1=T1, dt0=h, y0=y0, args=params, max_steps=None, adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=None))
    return solution.ys


# Dynamics being integrated backwards in time to compute the gradients in the backward pass
# Computed efficiently with a reverse-mode vector-jacobian product
def augmented_dynamics(t, s, params):
    z, a, _ = s
    d1, vjp_fun = jax.vjp(model_forward, z, params)
    d2, d3 = vjp_fun(a)
    d2 = -d2
    d3 = jtu.tree_map(lambda x: -x, d3)
    ds = (d1, d2, d3)
    return ds


augmented_term = diffrax.ODETerm(augmented_dynamics)


# Neural ODE backward pass
# Returns gradients from the output of the Neural ODE to both the parameters and the input
# The augmented state to be integrated is set as a tuple of three jax arrays
# Algorithm 1 from https://arxiv.org/abs/1806.07366
def backward(params, z_pred, output_grads):
    z1 = z_pred
    a1 = output_grads
    dL = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
    s1 = (z1, a1, dL)
    solution = diffrax.diffeqsolve(augmented_term, solver, t0=T1, t1=T0, dt0=-h, y0=s1, args=params, max_steps=None, adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=None))
    z0, input_grads, parameter_grads = solution.ys
    input_grads = input_grads[0, :]
    parameter_grads = jtu.tree_map(lambda x: jnp.squeeze(x), parameter_grads)
    return input_grads, parameter_grads


@jax.value_and_grad
def train_loss(y_pred, y_true):
    return jnp.mean(optax.l2_loss(y_pred, y_true))


# One step of the training algorithm
@jax.jit
def train(params, opt_state, y):
    y0 = y[0, :, :]
    y_pred = jax.vmap(forward, in_axes=(0, None))(y0, params)
    y_pred = jnp.squeeze(y_pred)
    y1 = y[-1, :, :]
    loss, loss_grads = train_loss(y_pred, y1)
    input_grads, parameter_grads = jax.vmap(backward, in_axes=(None, 0, 0))(params, y_pred, loss_grads)
    input_grads = jnp.mean(input_grads, axis=0)
    parameter_grads = jtu.tree_map(lambda x: jnp.mean(x, axis=0), parameter_grads)
    updates, opt_state = optimizer.update(parameter_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


def validation_loss(y_pred, y_true):
    return jnp.mean(optax.l2_loss(y_pred, y_true))


@jax.jit
def validate(params, y):
    y0 = y[0, :, :]
    y_pred = jax.vmap(forward, in_axes=(0, None))(y0, params)
    y_pred = jnp.squeeze(y_pred)
    y1 = y[-1, :, :]
    loss = validation_loss(y_pred, y1)
    return loss


def data_split(y, ratio=0.8):
    batch_size = y.shape[1]
    y_train = y[:, :int(batch_size * ratio), :]
    y_val = y[:, int(batch_size * ratio):, :]
    return y_train, y_val


def plot_vector_field(params):
    X = jnp.arange(-10, 10, 0.1)
    X1, X2 = jnp.meshgrid(X, X, indexing="xy")
    XX = jnp.stack((X1.flatten(), X2.flatten()), axis=1)

    YY = model_forward(XX, params)
    # YY = mass_spring_damper(None, XX, [1, 1, 1])

    Y1 = YY[:, 0].reshape(X1.shape)
    Y2 = YY[:, 1].reshape(X2.shape)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y1 = np.asarray(Y1)
    Y2 = np.asarray(Y2)

    plt.figure()
    plt.streamplot(X1, X2, Y1, Y2, density=1, linewidth=None, color="#A23BEC") 
    plt.show()


epochs = 0
pbar = tqdm(range(1, epochs + 1))
for epoch in pbar:
    try:
        key, subkey = jax.random.split(key)
        y = generate_data(data_term, data_args, dim, batch_size, subkey, solver, T0, T1, h, saveat)
        y_train, y_val = data_split(y)

        train_loss, params, opt_state = train(params, opt_state, y_train)
        val_loss = validate(params, y_val)

        pbar.set_postfix({"TL": f"{train_loss.item():.4f}", "VL": f"{val_loss.item():.4f}"})
    except KeyboardInterrupt:
        break

plot_vector_field(params)
