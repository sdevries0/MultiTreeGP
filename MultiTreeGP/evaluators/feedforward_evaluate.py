import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from typing import Tuple, Callable
import copy

class Evaluator:
    def __init__(self, env, dt0: float, solver=diffrax.Euler(), max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        """Evaluator for static symbolic policies in control tasks

        Attributes:
            env: Environment on which the candidate is evaluated
            max_fitness: Max fitness which is assigned when a trajectory returns an invalid value
            state_size: Dimensionality of the hidden state
            obs_size: Dimensionality of the observations
            control_size: Dimensionality of the control
            latent_size: Dimensionality of the state of the environment
            dt0: Initial step size for integration
            solver: Solver used for integration
            max_steps: The maximum number of steps that can be used in integration
            stepsize_controller: Controller for the stepsize during integration
        """
        self.env = env
        self.max_fitness = 1e4
        self.obs_size = env.n_obs
        self.control_size = env.n_control
        self.latent_size = env.n_var*env.n_dim
        self.dt0 = dt0
        self.solver = solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller

    def __call__(self, coefficients: Array, nodes: Array, data: Tuple, tree_evaluator: Callable) -> float:
        """Evaluates the candidate on a task

        :param coefficients: The coefficients of the candidate
        :param nodes: The nodes and index references of the candidate
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees

        Returns: Fitness of the candidate
        """
        _, _, _, fitness = self.evaluate_candidate(jnp.concatenate([nodes, coefficients], axis=-1), data, tree_evaluator)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_candidate(self, candidate: Array, data: Tuple, eval) -> Tuple[Array, Array, Array, float]:
        """Evaluates a candidate given a task and data

        :param candidate: Candidate that is evaluated
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees
        
        Returns: Predictions and fitness of the candidate
        """
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0, None])(candidate, *data, eval)
    
    def evaluate_control_loop(self, candidate: Array, x0: Array, ts: Array, target: Array, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, float]:
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation 
        of the control is defined by the set of trees

        :param candidate: Candidate with trees for the hidden state and readout
        :param x0: Initial state of the system
        :param ts: time points on which the controller is evaluated
        :param target: Target position that the system should reach
        :param process_noise_key: Key to generate process noise
        :param obs_noise_key: Key to generate noisy observations
        :param params: Parameters that define the environment
        :param tree_evaluator: Function for evaluating trees

        Returns: States, observations, control, activities of the hidden state of the candidate and the fitness of the candidate.
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        policy = candidate
        
        solver = self.solver
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)

        system = diffrax.ODETerm(self._drift)
        
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps, event=diffrax.Event(self.env.cond_fn_nan), 
            args=(env, policy, obs_noise_key, target, tree_evaluator), stepsize_controller=self.stepsize_controller, throw=False
        )

        xs = sol.ys
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        us = jax.vmap(lambda y, tar: tree_evaluator(policy, jnp.concatenate([y, tar])), in_axes=[0,None])(ys, target)

        fitness = env.fitness_function(xs, us, target, ts)

        return xs, ys, us, fitness
    
    #Define state equation
    def _drift(self, t, x, args):
        env, policy, obs_noise_key, target, tree_evaluator = args
        _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
        u = tree_evaluator(policy, jnp.concatenate([y, target]))

        dx = env.drift(t, x, u) #Apply control to system and get system change
        return dx
    
    #Define diffusion
    def _diffusion(self, t, x, args):
        env, policy, obs_noise_key, target, tree_evaluator = args

        return env.diffusion(t, x, jnp.array([0]))