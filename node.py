import random
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, NoAdjoint


seed = 42069
random.seed(seed)
np.random.seed(seed)
key = jax.random.PRNGKey(seed)


def mass_spring_damper(t, y, args):
    m, d, k = args
    A = jnp.array([[0, 1], [-k / m, -d / m]])
    return y @ A


# Generate trajectories by sampling random initial values and integrating those
def generate_data(data_term, dim, batch_size, key, T0, T1, h, saveat, solver):
    y0 = jax.random.ball(key, d=dim, p=2, shape=(batch_size,))
    solution = diffeqsolve(data_term, solver, t0=T0, t1=T1, dt0=h, y0=y0, saveat=saveat, args=[1, 1, 1], adjoint=NoAdjoint())
    return solution.ys


T0 = 0
T1 = 1
h = 0.01
saveat = SaveAt(ts=jnp.arange(T0, T1, h))
solver = Dopri5()

data_term = ODETerm(mass_spring_damper)
dim = 2
batch_size = 200


# Define models as a list of nodes per layer
# Return the parameters as a single one-dimensional vector
def model_init(model_def, key):
    parameter_count = 0
    for i in range(len(model_def) - 1):
        parameter_count += model_def[i] * model_def[i + 1] + model_def[i + 1]

    params = jax.random.normal(key, (parameter_count,))
    return params


key, subkey = jax.random.split(key)
model_def = [2, 50, 50, 2]
params = model_init(model_def, subkey)

optimizer = optax.adamw(learning_rate=1e-2)
opt_state = optimizer.init(params)


# Use the model definition to extract the individual parameters from the total vector
# The forward pass is computed as a standard fully connected neural network
# The sigmoid activation function is applied at every layer except the last
def model_forward(x, params):
    param_index = 0
    for i in range(len(model_def) - 1):
        layer_index = model_def[i] * model_def[i + 1]
        weights = params[param_index:param_index + layer_index]
        weights = jnp.reshape(weights, (model_def[i], model_def[i + 1]))

        param_index += layer_index
        layer_index = model_def[i + 1]
        bias = params[param_index:param_index + layer_index]

        param_index += layer_index

        x = x @ weights + bias
        if i < len(model_def) - 1:
            x = jax.nn.sigmoid(x)
    return x


# Vectorize the forward pass using vmap
# Can not use a decorator because the in_axes must be specified
model_forward = jax.vmap(model_forward, in_axes=(0, None))
    
model_term = ODETerm(lambda t, y, args: model_forward(y, args[0]))


@jax.value_and_grad
def compute_loss(y_pred, y_true):
    return jnp.mean(optax.l2_loss(y_pred, y_true))


# Neural ODE forward pass
def forward(y0, params):
    solution = diffeqsolve(model_term, solver, t0=T0, t1=T1, dt0=h, y0=y0, args=[params], adjoint=NoAdjoint())
    return solution.ys[-1, :, :]


# Dynamics being integrated backwards in time to compute the gradients in the backward pass
# The final vector-jacobian-product is currently not vectorized properly,
# which is (inefficiently) solved by duplicating and averaging the result
def augmented_dynamics(t, s, args):
    params = args[0]
    n = (s.shape[1] - params.shape[0]) // 2
    z = s[:, 0:n]
    a = s[:, n:2*n]

    batch_size = s.shape[0]

    # _, vjp_fun = jax.vjp(lambda theta: model_forward(z, theta), params)
    # d3 = vjp_fun(a)[0]
    # d3 = jnp.tile(d3, (batch_size, 1)) / batch_size

    # theta = jnp.tile(params, (batch_size, 1))

    # d1, vjp_fun = jax.vjp(lambda z, theta: jax.vmap(model_forward)(z, theta), z, theta)
    d1, vjp_fun = jax.vjp(model_forward, z, params)
    d2, d3 = vjp_fun(a)
    d3 = jnp.tile(d3, (batch_size, 1)) / batch_size
    ds = jnp.concatenate((d1, -d2, -d3), axis=1)
    return ds


augmented_term = ODETerm(augmented_dynamics)


# Neural ODE backward pass
# Returns gradients from the output of the Neural ODE to both the parameters and the input
# Algorithm 1 from https://arxiv.org/abs/1806.07366
def backward(params, z_pred, output_grads):
    batch_size, n = z_pred.shape
    z1 = z_pred
    a1 = output_grads
    dL = jnp.zeros((batch_size, params.shape[0]))
    s0 = jnp.concatenate((z1, a1, dL), axis=1)
    solution = diffeqsolve(augmented_term, solver, t0=T1, t1=T0, dt0=-h, y0=s0, args=[params], adjoint=NoAdjoint())
    augmented_state = solution.ys[-1, :, :]
    input_grads = jnp.mean(augmented_state[:, n:2 * n], axis=0)
    parameter_grads = jnp.mean(augmented_state[:, 2 * n:], axis=0)
    return input_grads, parameter_grads


# One step of the training algorithm
@jax.jit
def train(params, opt_state, y):
    y0 = y[0, :, :]
    y_pred = forward(y0, params)
    y1 = y[-1, :, :]
    loss, loss_grads = compute_loss(y_pred, y1)
    input_grads, parameter_grads = backward(params, y_pred, loss_grads)
    updates, opt_state = optimizer.update(parameter_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


def validation_loss(y_pred, y_true):
    return jnp.mean(optax.l2_loss(y_pred, y_true))


@jax.jit
def validate(params, y):
    y0 = y[0, :, :]
    y_pred = forward(y0, params)
    y1 = y[-1, :, :]
    loss = validation_loss(y_pred, y1)
    return loss


def data_split(y, ratio=0.8):
    batch_size = y.shape[1]
    y_train = y[:, :int(batch_size * ratio), :]
    y_val = y[:, int(batch_size * ratio):, :]
    return y_train, y_val


epochs = 100
for epoch in range(1, epochs + 1):
    try:
        key, subkey = jax.random.split(key)
        y = generate_data(data_term, dim, batch_size, subkey, T0, T1, h, saveat, solver)
        y_train, y_val = data_split(y)

        train_loss, params, opt_state = train(params, opt_state, y_train)
        val_loss = validate(params, y_val)

        print(f"Epoch: {epoch:3d}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    except KeyboardInterrupt:
        break


def plot_vector_field(params):
    x = np.arange(-10, 10, 0.1)
    n = x.shape[0]

    X = np.zeros((n * n, dim))
    for i in range(n):
        for j in range(n):
            X[i + n * j, 0] = x[j]
            X[i + n * j, 1] = x[i]
    
    XX = jnp.asarray(X)
    YY = model_forward(XX, params)
    Y = np.asarray(YY)

    X1 = np.zeros((n, n))
    X2 = np.zeros((n, n))
    Y1 = np.zeros((n, n))
    Y2 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            X1[i, j] = X[i + n * j, 0]
            X2[i, j] = X[i + n * j, 1]
            Y1[i, j] = Y[i + n * j, 0]
            Y2[i, j] = Y[i + n * j, 1]

    plt.figure()
    plt.streamplot(X1, X2, Y1, Y2, density=1, linewidth=None, color="#A23BEC")
    plt.show()

# plot_vector_field(params)
