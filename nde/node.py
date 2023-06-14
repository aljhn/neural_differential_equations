import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import diffrax


class NODE:
    """
    Neural Ordinary Differential Equation (Neural ODE)
    """

    def __init__(self, model_forward, T0, T1, h, solver=None):
        self.model_forward = model_forward
        self.model_term = diffrax.ODETerm(lambda t, y, args: model_forward(y, args))
        if solver is None:
            self.solver = diffrax.Dopri5()
        else:
            self.solver = solver
        self.T0 = T0
        self.T1 = T1
        self.h = h
        self.augmented_term = diffrax.ODETerm(self.augmented_dynamics)

    def forward(self, y0, params):
        """
        Neural ODE forward pass.
        Use the input as an initial value and integrate it using the model as
        the dynamics. Using the parameters max_steps=None and
        adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=None) means that
        diffrax disables automatic differentiation, as this is implemented manually
        with the augmented dynamics.
        """
        solution = diffrax.diffeqsolve(
            self.model_term,
            self.solver,
            t0=self.T0,
            t1=self.T1,
            dt0=self.h,
            y0=y0,
            args=params,
            max_steps=None,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=None),
        )
        return solution.ys

    def augmented_dynamics(self, t, s, params):
        """
        Dynamics being integrated backwards in time to compute the gradients in the
        backward pass. Computed efficiently with a reverse-mode vector-jacobian product.
        """
        z, a, _ = s
        d1, vjp_fun = jax.vjp(self.model_forward, z, params)
        d2, d3 = vjp_fun(a)
        d2 = -d2
        d3 = jtu.tree_map(lambda x: -x, d3)
        ds = (d1, d2, d3)
        return ds

    def backward(self, params, z_pred, output_grads):
        """
        Neural ODE backward pass.
        Returns gradients from the output of the Neural ODE to both the parameters
        and the input. The augmented state to be integrated is set as a tuple of
        three jax arrays.
        Algorithm 1 from https://arxiv.org/abs/1806.07366

        """
        z1 = z_pred
        a1 = output_grads
        dL = jtu.tree_map(lambda x: jnp.zeros_like(x), params)
        s1 = (z1, a1, dL)
        solution = diffrax.diffeqsolve(
            self.augmented_term,
            self.solver,
            t0=self.T1,
            t1=self.T0,
            dt0=-self.h,
            y0=s1,
            args=params,
            max_steps=None,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=None),
        )
        z0, input_grads, parameter_grads = solution.ys
        input_grads = input_grads[0, :]
        parameter_grads = jtu.tree_map(lambda x: jnp.squeeze(x), parameter_grads)
        return input_grads, parameter_grads
