import jax
from jax import Array
import jax.numpy as jnp
from typing import Tuple
import diffrax
import optimistix as optx
import time
from functools import partial

class Evaluator:
    """Evaluator for symbolic expressions on symbolic regression tasks.

    Attributes:
        max_fitness: Max fitness which is assigned when a trajectory returns an invalid value.
        dt0: Step size for solve.
        fitness_function: Function that computes the fitness of a candidate
    """
    def __init__(self, solver: diffrax.AbstractSolver = diffrax.Euler(), dt0: float = 0.01, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        self.max_fitness = 1e5
        self.dt0 = dt0
        self.fitness_function = lambda pred_ys, true_ys: jnp.mean(jnp.sum(jnp.square(pred_ys-true_ys), axis=-1)) #Mean Squared Error
        self.system = diffrax.ODETerm(self._drift)
        self.solver = solver
        self.stepsize_controller = stepsize_controller

    def __call__(self, model: jnp.ndarray, data: Tuple, eval) -> float:
        """Evaluates the model on an environment.

        :param model: Callable model of symbolic expressions.
        :param data: The data required to evaluate the model.
        :returns: Fitness of the model.
        """
        # model = jnp.concatenate([left, right[:,:,None]], axis=2)
        fitness, _ = self.evaluate_model(model, data, eval)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_model(self, model: jnp.ndarray, data: Tuple, eval) -> Tuple[Array, float]:
        """Evaluate a tree by solving the model as a differential equation.

        :param model: Callable model of symbolic expressions.
        :param data: The data required to evaluate the model.
        :returns: Predictions and fitness of the model.
        """
        return jax.vmap(self.evaluate_time_series, in_axes=[None, 0, None, 0, None])(model, *data, eval)
    
    def evaluate_time_series(self, model: jnp.ndarray, x0: Array, ts: Array, ys: Array, eval) -> Tuple[Array, float]:
        """Solves the model to get predictions. The predictions are used to compute the fitness of the model.

        :param model: Callable model of symbolic expressions. 
        :param x0: Initial conditions of the environment.
        :param ts: Timepoints of which the system has to be solved.
        :param ys: Ground truth data used to compute the fitness.
        :returns: Predictions and fitness of the model.
        """
        
        saveat = diffrax.SaveAt(ts=ts)
        root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
        event_nan = diffrax.Event(self.cond_fn_nan)#, root_finder=root_finder) 

        sol = diffrax.diffeqsolve(
            self.system, self.solver, ts[0], ts[-1], self.dt0, x0, args=(model, eval), saveat=saveat, max_steps=16**4, stepsize_controller=self.stepsize_controller
        )

        fitness = self.fitness_function(sol.ys, ys)

        return fitness, sol.ys
    
    def _drift(self, t, x, args):
            model, eval = args
            dx = eval(model, x)
            return dx
    
    def cond_fn_nan(self, t, y, args, **kwargs):
        return jnp.where(jnp.any(jnp.isinf(y) +jnp.isnan(y)), -1.0, 1.0)