import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from typing import Tuple
import copy

class Evaluator:
    """Evaluator for feedforward symbolic policies in control tasks.

    Attributes:
        env: Environment on which the model is evaluated.
        max_fitness: Max fitness which is assigned when a trajectory returns an invalid value.
        latent_size: Dimensionality of the environment.
        dt0: Step size for solve.
    """
    def __init__(self, env, dt0: float) -> None:
        self.env = env
        self.max_fitness = 1e4
        self.latent_size = env.n_var
        self.dt0 = dt0

    def __call__(self, model, data) -> float:
        """Computes the fitness of a model.

        :param model: Model with trees for the hidden state and readout.
        :param data: The data required to evaluate the controller.

        Returns: The fitness of the model.
        """
        _, _, _, fitness = self.evaluate_model(model, data)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_model(self, model: Tuple, data: Tuple) -> Tuple[Array, Array, Array, float]:
        """Evaluate a tree by simulating the environment and controller as a coupled system.

        :param model: Model with trees for the hidden state and readout.
        :param data: The data required to evaluate the controller.

        Returns: States, observations, control and the fitness of the model.
        """
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0])(model, *data)
    
    def evaluate_control_loop(self, model: Tuple, x0: Array, ts: Array, target: Array, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple) -> Tuple[Array, Array, Array, float]:
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment.

        :param model: Model with trees for the hidden state and readout
        :param x0: Initial state of the system
        :param ts: time points on which the controller is evaluated
        :param target: Target position that the system should reach
        :param key: Random key.
        :param params: Parameters that define the environment.

        Returns: States, observations, control and the fitness of the model.
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        policy = model[0]

        #Define state equation
        def _drift(t, x, args):
            _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
            u = policy({"y":y, "tar":target}) #Readout control from hidden state

            dx = env.drift(t, x, u) #Apply control to system and get system change
            return dx
        
        #Define diffusion
        def _diffusion(t, x, args):
            _, y = env.f_obs(obs_noise_key, (t, x))
            u = policy({"y":y, "tar":target})
            # u = a
            return env.diffusion(t, x, u)
        
        solver = diffrax.EulerHeun()
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key, levy_area=diffrax.BrownianIncrement)
        system = diffrax.MultiTerm(diffrax.ODETerm(_drift), diffrax.ControlTerm(_diffusion, brownian_motion))
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=16**5, discrete_terminating_event=self.env.terminate_event
        )

        xs = sol.ys
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        us = jax.vmap(lambda y, tar: policy({"y":y, "tar":tar}), in_axes=[0,None])(ys, target)

        fitness = env.fitness_function(xs, us, target, ts)

        return xs, ys, us, fitness