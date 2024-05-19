import jax
from jax import Array
import jax.numpy as jnp
from jax.random import PRNGKey
from typing import Tuple
import copy
import diffrax

class Evaluator:
    """Evaluator for the LQG controller.

    Attributes:
        env: Environment in which the LQG acts.
        latent_size: Dimensionality of the environment.
        dt0: Step size for solve.
    """
    def __init__(self, env, dt0: float) -> None:
        self.env = env
        self.latent_size = env.n_var*env.n_dim
        self.dt0 = dt0

    def compute_ricatti(self, A: Array, b: Array, Q: Array, R: Array) -> Array:
        """Solves the Ricatti equations.

        :param A
        :param b
        :param Q
        :param R
        :returns: Ricatti matrix
        """
        def _f(t,x,args):
            return A.T@x + x@A - x@b@jnp.linalg.inv(R)@b.T@x + Q
        
        system = diffrax.ODETerm(_f)
        sol = diffrax.diffeqsolve(
            system, diffrax.Euler(), 0, 10, 0.01, jnp.zeros_like(Q), max_steps=16**4
        )
        return sol.ys[0]

    def evaluate_LQG(self, x0: Array, ts: Array, target: Array, process_noise_key: PRNGKey, obs_noise_key: PRNGKey, params: Tuple) -> Tuple[Array, Array, Array, Array]:
        """Evaluates the LQG controller in an environment.

        :param model: Callable network of symbolic expressions. 
        :param x0: Initial conditions of the environment.
        :param ts: Timepoints of which the system has to be solved.
        :param target: Target state to be reached.
        :param process_noise_key: Key to add stochasticity into the dynamics.
        :param obs_noise_key: Key to add noise to observations.
        :param params: Params that define the environment.
        :returns: State, observations, control and fitness of the simulation.
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        def drift(t, variables, args):
            x_star, u_star, L = args
            x = variables[:self.latent_size]
            mu = variables[self.latent_size:self.latent_size*2]
            P = variables[2*self.latent_size:].reshape(self.latent_size,self.latent_size)
            
            _, y = env.f_obs(obs_noise_key, (t, x))
            u = jnp.array(-L@(mu-x_star) + u_star)
            K = P@env.C.T@jnp.linalg.inv(env.W)

            dx = env.drift(t,x,u)
            dmu = env.A@mu + env.b@u + K@(y-env.C@mu)
            dP = env.A@P + P@env.A.T-K@env.C@P+env.V

            return jnp.concatenate([dx, dmu, jnp.ravel(dP)])

        #Apply process noise only on x
        def diffusion(t, variables, args):
            x = variables[:self.latent_size]
            return jnp.concatenate([env.diffusion(t,x,args),jnp.zeros((self.latent_size,self.latent_size)),jnp.zeros((self.latent_size**2,self.latent_size))])
        
        solver = diffrax.EulerHeun()
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)

        L = jnp.linalg.inv(env.R)@env.b.T@self.compute_ricatti(env.A, env.b, env.Q, env.R)

        #Set target state and control
        x_star = jnp.zeros((self.latent_size))
        for i in range(env.n_dim):
            x_star = x_star.at[i*env.n_var].set(target[i])
        u_star = -jnp.linalg.pinv(env.b)@env.A@x_star

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key) #define process noise
        system = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))

        init = jnp.concatenate([x0, env.mu0, jnp.ravel(env.P0)])

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, init, saveat=saveat, args=(x_star, u_star, L), adjoint=diffrax.DirectAdjoint(), max_steps=16**7
        )
        x = sol.ys[:,:self.latent_size]
        mu = sol.ys[:,self.latent_size:2*self.latent_size]
        u = jax.vmap(lambda m, l: -l@(m-x_star) + u_star, in_axes=[0,None])(mu, L) #Map state estimates to control
        _, y = jax.lax.scan(env.f_obs, obs_noise_key, (ts, x)) #Map states to observations

        costs = env.fitness_function(x, u, target, ts)

        return x, y, u, costs

    def __call__(self, data: Tuple) -> Tuple[Array, Array, Array, float]:
        """Evaluates the LQG controller in an environment.

        :param data: The data required to evaluate the controller.
        :returns: State, observations, control and fitness of the simulation.
        """
        x, y, u, costs = jax.vmap(self.evaluate_LQG, in_axes=[0, None, 0, 0, 0, 0])(*data)
        
        return x, y, u, jnp.mean(costs)