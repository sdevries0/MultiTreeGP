import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from typing import Sequence, Tuple, Callable
import copy

class Evaluator:
    def __init__(self, env, state_size: int, dt0: float, solver=diffrax.Euler(), max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        """Evaluator for dynamic symbolic policies in control tasks

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
        self.state_size = state_size
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
        _, _, _, _, fitness = self.evaluate_candidate(jnp.concatenate([nodes, coefficients], axis=-1), data, tree_evaluator)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_candidate(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, Array, float]:
        """Evaluates a candidate given a task and data

        :param candidate: Candidate that is evaluated
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees
        
        Returns: Predictions and fitness of the candidate
        """
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0, None])(candidate, *data, tree_evaluator)
    
    def evaluate_control_loop(self, candidate: Array, x0: Array, ts: Array, target: float, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, Array, float]:
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

        state_equation = candidate[:self.state_size]
        readout = candidate[self.state_size:]
        
        solver = self.solver
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        system = diffrax.ODETerm(self._drift)

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps, event=diffrax.Event(self.env.cond_fn_nan), 
            args=(env, state_equation, readout, obs_noise_key, target, tree_evaluator), stepsize_controller=self.stepsize_controller, throw=False
        )

        xs = sol.ys[:,:self.latent_size]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        activities = sol.ys[:,self.latent_size:]
        us = jax.vmap(lambda y, a, tar: tree_evaluator(readout, jnp.concatenate([y, a, jnp.zeros(self.control_size), target])), in_axes=[0,0,None])(ys, activities, target)

        fitness = env.fitness_function(xs, us, target, ts)

        return xs, ys, us, activities, fitness
    
    def _drift(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, target, tree_evaluator = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]

        _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
        u = tree_evaluator(readout, jnp.concatenate([jnp.zeros(self.obs_size), a, jnp.zeros(self.control_size), target]))

        dx = env.drift(t, x, u) #Apply control to system and get system change
        da = tree_evaluator(state_equation, jnp.concatenate([y, a, u, target]))

        return jnp.concatenate([dx, da])
    
    def _diffusion(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, target, tree_evaluator = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]

        return jnp.concatenate([env.diffusion(t, x, jnp.array([0])), jnp.zeros((self.state_size, self.latent_size))]) #Only the system is stochastic
    

class EvaluatorMT:
    def __init__(self, env, state_size, dt0) -> None:
        self.env = env
        self.max_fitness = 1e4
        self.state_size = state_size
        self.latent_size = env.n_var*env.n_dim
        self.dt0 = dt0

    def __call__(self, candidate, data) -> float:
        _, _, _, _, fitness = self.evaluate_candidate(candidate, data)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_candidate(self, candidate, data):
        "Evaluate a tree by simulating the environment and controller as a coupled system"
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0])(candidate, *data)
    
    def evaluate_control_loop(self, candidate: Tuple, x0: Sequence[float], ts: Sequence[float], target: float, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple):
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation 
        of the control is defined by the set of trees
        Inputs:
            candidate (NetworkTrees): Candidate with trees for the hidden state and readout
            x0 (float): Initial state of the system
            ts (Array[float]): time points on which the controller is evaluated
            target (float): Target position that the system should reach
            key (PRNGKey)
            params (Tuple[float]): Parameters that define the system

        Returns:
            xs (Array[float]): States of the system at every time point
            ys (Array[float]): Observations of the system at every time point
            us (Array[float]): Control of the candidate at every time point
            activities (Array[float]): Activities of the hidden state of the candidate at every time point
            fitness (float): Fitness of the candidate 
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        state_equation, readout = candidate

        #Define state equation
        def _drift(t, x_a, args):
            tar = args.evaluate(t)
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]

            _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
            u = readout({"a":a, "tar":jnp.array([tar])}) #Readout control from hidden state

            dx = env.drift(t, x, u) #Apply control to system and get system change
            da = state_equation({"y":y, "a":a, "u":u, "tar":jnp.array([tar])}) #Compute hidden state updates
            return jnp.concatenate([dx, da])
        
        #Define diffusion
        def _diffusion(t, x_a, args):
            tar = args.evaluate(t)
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]
            _, y = env.f_obs(obs_noise_key, (t, x))
            u = readout({"a":a, "tar":jnp.array([tar])})
            # u = a
            return jnp.concatenate([env.diffusion(t, x, u), jnp.zeros((self.state_size, self.latent_size))]) #Only the system is stochastic
        
        a = target[0]*jnp.ones(int(ts.shape[0]//4))
        b = target[1]*jnp.ones(int(ts.shape[0]//4))
        c = target[2]*jnp.ones(int(ts.shape[0]//4))
        d = target[3]*jnp.ones(int(ts.shape[0]//4))
        targets = diffrax.LinearInterpolation(ts, jnp.hstack((a,b,c,d)))

        solver = diffrax.EulerHeun()
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key, levy_area=diffrax.BrownianIncrement)
        system = diffrax.MultiTerm(diffrax.ODETerm(_drift), diffrax.ControlTerm(_diffusion, brownian_motion))
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat, args=targets, adjoint=diffrax.DirectAdjoint(), max_steps=16**5, discrete_terminating_event=self.env.terminate_event
        )

        xs = sol.ys[:,:self.latent_size]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        activities = sol.ys[:,self.latent_size:]
        us = jax.vmap(lambda a, t, tar: readout({"a":a, "tar":jnp.array([tar.evaluate(t)])}), in_axes=[0,0,None])(activities, ts, targets)

        # fitness = env.fitness_function(xs, us, targets, ts)

        return xs, ys, us, activities, 0