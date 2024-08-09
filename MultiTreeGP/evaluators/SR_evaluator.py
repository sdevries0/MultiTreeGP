import jax
from jax import Array
import jax.numpy as jnp
from typing import Tuple
import diffrax
import time

class Evaluator:
    """Evaluator for symbolic expressions on symbolic regression tasks.

    Attributes:
        max_fitness: Max fitness which is assigned when a trajectory returns an invalid value.
        dt0: Step size for solve.
        fitness_function: Function that computes the fitness of a candidate
    """
    def __init__(self, dt0: float) -> None:
        self.max_fitness = 1e5
        self.dt0 = dt0
        self.fitness_function = lambda pred_ys, true_ys: jnp.mean(jnp.sum(jnp.square(pred_ys-true_ys), axis=-1)) #Mean Squared Error

    def __call__(self, model: Tuple, data: Tuple) -> float:
        """Evaluates the model on an environment.

        :param model: Callable model of symbolic expressions.
        :param data: The data required to evaluate the model.
        :returns: Fitness of the model.
        """
        _, fitness = self.evaluate_model(model, data)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_model(self, model: Tuple, data: Tuple) -> Tuple[Array, float]:
        """Evaluate a tree by solving the model as a differential equation.

        :param model: Callable model of symbolic expressions.
        :param data: The data required to evaluate the model.
        :returns: Predictions and fitness of the model.
        """
        return jax.vmap(self.evaluate_time_series, in_axes=[None, 0, None, 0])(model, *data)
    
    def evaluate_time_series(self, model: Tuple, x0: Array, ts: Array, ys: Array) -> Tuple[Array, float]:
        """Solves the model to get predictions. The predictions are used to compute the fitness of the model.

        :param model: Callable model of symbolic expressions. 
        :param x0: Initial conditions of the environment.
        :param ts: Timepoints of which the system has to be solved.
        :param ys: Ground truth data used to compute the fitness.
        :returns: Predictions and fitness of the model.
        """

        model = model[0]

        #Define state equation
        def _drift(t, x, args):
            dx = model(data=x)
            return dx
        
        solver = diffrax.Euler()
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)

        system = diffrax.ODETerm(_drift)
        start = time.time()
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, x0, saveat=saveat, max_steps=16**4
        )
        print("end", time.time() - start)

        fitness = self.fitness_function(sol.ys, ys)

        return sol.ys, fitness