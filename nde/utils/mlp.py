import jax
import jax.numpy as jnp


def mlp_init(input_dim, output_dim, hidden_dim, hidden_layers, key):
    """
    Create the parameters for a MLP with the given specification.
    The parameters are represented as a list of dictionaries,
    containing weights and biases.
    """
    model_def = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
    subkeys = jax.random.split(key, num=(len(model_def) - 1) * 2)
    params = []
    for i in range(len(model_def) - 1):
        layer = {
            "weights": jax.random.normal(subkeys[i], (model_def[i], model_def[i + 1])),
            "bias": jax.random.normal(
                subkeys[i + len(model_def) - 1], (model_def[i + 1],)
            ),
        }
        params.append(layer)
    return params


def mlp_forward(x, params, activation_function=jnp.tanh):
    """
    Standard MLP forward pass using the given input x and params.
    """
    for i in range(len(params)):
        weights = params[i]["weights"]
        bias = params[i]["bias"]
        x = x @ weights + bias
        if i < len(params) - 1:
            x = activation_function(x)
    return x
