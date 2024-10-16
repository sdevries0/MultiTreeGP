import jax
from jax import Array
import jax.numpy as jnp
from typing import Tuple, Callable
import diffrax
import optimistix as optx
from jax.random import PRNGKey

class Evaluator:
    """Evaluator for symbolic expressions on symbolic regression tasks.

    Attributes:
        max_fitness: Max fitness which is assigned when a trajectory returns an invalid value.
        dt0: Step size for solve.
        fitness_function: Function that computes the fitness of a candidate
    """
    def __init__(self, solver: diffrax.AbstractSolver = diffrax.Euler(), dt0: float = 0.01, max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        self.max_fitness = 1e5
        self.dt0 = dt0
        self.fitness_function = lambda pred_ys, true_ys: jnp.mean(jnp.sum(jnp.square(pred_ys-true_ys), axis=-1)) #Mean Squared Error
        self.system = diffrax.ODETerm(self._drift)
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.max_steps = max_steps

    def __call__(self, coefficients: Array, nodes: Array, data: Tuple, tree_evaluator: Callable) -> float:
        
        """Evaluates the candidate on a task

        :param coefficients: The coefficients of the candidate
        :param nodes: The nodes and index references of the candidate
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees

        Returns: Fitness of the model
        """
        fitness, _ = self.evaluate_candidate(jnp.concatenate([nodes, coefficients], axis=-1), data, tree_evaluator)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_candidate(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> Tuple[Array, float]:
        """Evaluates a candidate given a task and data

        :param candidate: Candidate that is evaluated
        :param data: The data required to evaluate the candidate
        
        Returns: Predictions and fitness of the candidate
        """
        return jax.vmap(self.evaluate_time_series, in_axes=[None, 0, None, 0, 0, None])(candidate, *data, tree_evaluator)
    
    def evaluate_time_series(self, candidate: Array, x0: Array, ts: Array, ys: Array, process_noise_key: PRNGKey, tree_evaluator: Callable) -> Tuple[Array, float]:
        """Solves the candidate as a differential equation and returns the predictions and fitness

        :param candidate: Candidate that is evaluated
        :param x0: Initial conditions of the environment
        :param ts: Timepoints of which the system has to be solved
        :param ys: Ground truth data used to compute the fitness
        :param process_noise_key: Key to generate process noise
        :param tree_evaluator: Function for evaluating trees
        
        Returns: Predictions and fitness of the candidate
        """
        
        saveat = diffrax.SaveAt(ts=ts)
        root_finder = optx.Newton(1e-5, 1e-5, optx.rms_norm)
        event_nan = diffrax.Event(self.cond_fn_nan)#, root_finder=root_finder)
        # brownian_motion = diffrax.UnsafeBrownianPath(shape=(x0.shape[0],), key=process_noise_key, levy_area=diffrax.SpaceTimeLevyArea)
        # system = diffrax.MultiTerm(self.system, diffrax.ControlTerm(self._diffusion, brownian_motion))

        sol = diffrax.diffeqsolve(
            self.system, self.solver, ts[0], ts[-1], self.dt0, x0, args=(candidate, tree_evaluator), saveat=saveat, max_steps=self.max_steps, stepsize_controller=self.stepsize_controller, 
            adjoint=diffrax.DirectAdjoint(), throw=False, event=event_nan
        )
        pred_ys = sol.ys
        fitness = self.fitness_function(pred_ys, ys)

        return fitness, pred_ys
    
    def _drift(self, t, x, args):
        candidate, tree_evaluator = args
        dx = tree_evaluator(candidate, x)
        return dx
    
    def _diffusion(self, t, x, args):
        return jnp.zeros_like(x)
    
    def cond_fn_nan(self, t, y, args, **kwargs):
        return jnp.where(jnp.any(jnp.isinf(y) +jnp.isnan(y)), -1.0, 1.0)