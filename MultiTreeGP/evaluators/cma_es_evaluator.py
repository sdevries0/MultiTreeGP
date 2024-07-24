import jax
from jax import Array
import jax.numpy as jnp
from jax.random import PRNGKey
import diffrax

from typing import Tuple
import copy

from MultiTreeGP.networks.NDE import NDE, ParameterReshaper

class Evaluator:
    """Evaluator for NDE policies in control tasks.

        Attributes:
            env: Environment on which the model is evaluated.
            max_fitness: Max fitness which is assigned when a trajectory returns an invalid value.
            state_size: Dimensionality of the hidden state.
            latent_size: Dimensionality of the environment.
            dt0: Step size for solve.
            parameter_reshaper: Maps array of parameters to the structure of the NDE.
        """
    def __init__(self, env, parameter_reshaper: ParameterReshaper, state_size: int, dt0: float) -> None:
        self.env = env
        self.max_fitness = 1e4
        self.state_size = state_size
        self.latent_size = self.env.n_var*self.env.n_dim
        self.dt0 = dt0
        self.parameter_reshaper = parameter_reshaper

    def evaluate_control_loop(self, model: NDE, x0: Array, ts: Array, target: float, process_noise_key: PRNGKey, obs_noise_key: PRNGKey, params: Tuple) -> Tuple[Array, Array, Array, Array, float]:
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation 
        of the control is defined by the set of trees

        :param model: Model with trees for the hidden state and readout
        :param x0: Initial state of the system
        :param ts: time points on which the controller is evaluated
        :param target: Target position that the system should reach
        :param key: Random key.
        :param params: Parameters that define the environment.

        Returns: States, observations, control, activities of the hidden state of the model and the fitness of the model.
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        #Define state equation
        def _drift(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]

            _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
            u = model.act(jnp.concatenate([a, target]))

            # u = a
            dx = env.drift(t, x, u) #Apply control to system and get system change
            da = model.update(jnp.concatenate([y, u, target]), a) #Compute hidden state updates

            return jnp.concatenate([dx, da])
        
        #Define diffusion
        def _diffusion(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]
            _, y = env.f_obs(obs_noise_key, (t, x))
            
            u = model.act(jnp.concatenate([a, target]))
            return jnp.concatenate([env.diffusion(t, x, u), jnp.zeros((self.state_size, self.latent_size))]) #Only the system is stochastic
        
        solver = diffrax.EulerHeun()
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key, levy_area=diffrax.BrownianIncrement)
        system = diffrax.MultiTerm(diffrax.ODETerm(_drift), diffrax.ControlTerm(_diffusion, brownian_motion))

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=500000, discrete_terminating_event=env.terminate_event
        )

        xs = sol.ys[:,:self.latent_size]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        activities = sol.ys[:,self.latent_size:]
        
        us = jax.vmap(lambda a, tar: model.act(jnp.concatenate([a, tar])), in_axes=[0,None])(activities, target)

        fitness = env.fitness_function(xs, us, target, ts)

        return xs, ys, us, activities, fitness

    def __call__(self, weights: Array, data: Tuple) -> float:
        """Computes the fitness of a model.

        :param model: Model with trees for the hidden state and readout.
        :param data: The data required to evaluate the controller.

        Returns: The fitness of the model.
        """
        _, _, _, _, fitness = self.evaluate_model(weights, data)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
         
    def evaluate_model(self, weights: Array, data: Tuple) -> Tuple[Array, Array, Array, Array, float]:
        """Evaluate a tree by simulating the environment and controller as a coupled system.

        :param model: Model with trees for the hidden state and readout.
        :param data: The data required to evaluate the controller.

        Returns: States, observations, control, activities of the hidden state of the model and the fitness of the model.
        """        
        model = NDE(*self.parameter_reshaper(weights))

        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0])(model, *data)