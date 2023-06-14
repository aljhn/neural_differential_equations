import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_vector_field(vector_field_function, width=10, step=0.1):
    """
    Plot the given vector field as a streamplot. Only works for two-dimensional vector fields.

    Args:
        vector_field_function: Function representing the vector field. Takes in a single argument with shape (N, 2).
        width: Size of the grid to visualize.
        step: Level of detail in the grid to visualize.

    Returns:
        None
    """
    X = jnp.arange(-width, width, step)
    X1, X2 = jnp.meshgrid(X, X, indexing="xy")
    XX = jnp.stack((X1.flatten(), X2.flatten()), axis=1)

    YY = vector_field_function(XX)

    Y1 = YY[:, 0].reshape(X1.shape)
    Y2 = YY[:, 1].reshape(X2.shape)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y1 = np.asarray(Y1)
    Y2 = np.asarray(Y2)

    plt.figure()
    plt.streamplot(X1, X2, Y1, Y2, density=1, linewidth=None, color="#A23BEC")
    plt.tight_layout()
    plt.show()
