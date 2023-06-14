import jax
import jax.numpy as jnp
import diffrax


def generate_data(
    dynamics_function,
    dynamics_args,
    dim,
    width,
    batch_size,
    key,
    T0,
    T1,
    h,
    saveat=None,
    solver=None,
):
    """
    Generate trajectories by sampling random initial values
    and integrating those in time.
    """
    if saveat is None:
        saveat = diffrax.SaveAt(ts=jnp.array([T0, T1]))
    if solver is None:
        solver = diffrax.Dopri5()

    y0 = jax.random.ball(key, d=dim, p=2, shape=(batch_size,)) * width
    term = diffrax.ODETerm(dynamics_function)
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=T0,
        t1=T1,
        dt0=h,
        y0=y0,
        saveat=saveat,
        args=dynamics_args,
        max_steps=None,
        adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=None),
    )
    return solution.ys


def data_split(y, ratio=0.8):
    batch_size = y.shape[1]
    y_train = y[:, : int(batch_size * ratio), :]
    y_val = y[:, int(batch_size * ratio) :, :]
    return y_train, y_val


def mass_spring_damper(t, y, args):
    """
    Mass spring damper ODE.
    """
    m, d, k = args
    A = jnp.array([[0, 1], [-k / m, -d / m]])
    return y @ A.T
