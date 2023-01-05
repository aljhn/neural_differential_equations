import random
import numpy as np
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


data_term = ODETerm(mass_spring_damper)
solver = Dopri5()

T0 = 0
T1 = 1
h = 0.01
saveat = SaveAt(ts=jnp.arange(T0, T1, h))

batch_size = 200

key, subkey = jax.random.split(key)
y0 = jax.random.ball(subkey, d=2, p=2, shape=(batch_size,))
solution = diffeqsolve(data_term, solver, t0=T0, t1=T1, dt0=h, y0=y0, saveat=saveat, args=[1, 1, 1], adjoint=NoAdjoint())
y = solution.ys

model_def = [2, 50, 50, 2]
parameter_count = 0
for i in range(len(model_def) - 1):
    parameter_count += model_def[i] * model_def[i + 1] + model_def[i + 1]

key, subkey = jax.random.split(key)
params = jax.random.normal(subkey, (parameter_count,))

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)


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
model_forward = jax.vmap(model_forward, in_axes=(0, None))
    
model_term = ODETerm(lambda t, y, args: model_forward(y, args[0]))


@jax.value_and_grad
def compute_loss(y_pred, y):
    return jnp.mean(optax.l2_loss(y_pred, y))


def forward(y0, params):
    solution = diffeqsolve(model_term, solver, t0=T0, t1=T1, dt0=h, y0=y0, args=[params], adjoint=NoAdjoint())
    y_pred = solution.ys[-1, :, :]
    return y_pred


def augmented_dynamics(t, s, args):
    n = (s.shape[1] - parameter_count) // 2
    params = args[0]
    z = s[:, 0:n]
    a = s[:, n:2*n]

    d1, d2 = jax.jvp(lambda x: model_forward(x, params), (z,), (a,))

    _, vjp_fun = jax.vjp(lambda theta: model_forward(z, theta), params)
    d3 = vjp_fun(a)[0]
    d3 = jnp.tile(d3, (batch_size, 1)) / batch_size
    ds = jnp.concatenate((d1, -d2, -d3), axis=1)
    return ds


augmented_term = ODETerm(augmented_dynamics)


def backward(params, z_pred, loss_grad):
    n = z_pred.shape[1]
    z1 = z_pred
    a1 = loss_grad
    dL = jnp.zeros((batch_size, parameter_count))
    s0 = jnp.concatenate((z1, a1, dL), axis=1)
    solution = diffeqsolve(augmented_term, solver, t0=T1, t1=T0, dt0=-h, y0=s0, args=[params], adjoint=NoAdjoint())
    augmented_state = solution.ys[-1, :, :]
    input_grads = jnp.mean(augmented_state[:, n:2*n], axis=0)
    parameter_grads = jnp.mean(augmented_state[:, -parameter_count:], axis=0)
    return input_grads, parameter_grads


@jax.jit
def step(params, opt_state, y0, y):
    y_pred = forward(y0, params)
    y1 = y[-1, :, :]
    loss, loss_grads = compute_loss(y_pred, y1)
    input_grads, parameter_grads = backward(params, y_pred, loss_grads)
    updates, opt_state = optimizer.update(parameter_grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


epochs = 100
for epoch in range(1, epochs + 1):
    loss, params, opt_state = step(params, opt_state, y0, y)
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
