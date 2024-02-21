import jax
import jax.numpy as jnp
import time
from typing import Sequence, Tuple
import copy
import diffrax

class Evaluator:
    def __init__(self, env):
        self.env = env

    def compute_ricatti(self, A, b, Q, R):
        def _f(t,x,args):
            return A.T@x + x@A - x@b@jnp.linalg.inv(R)@b.T@x + Q
        
        system = diffrax.ODETerm(_f)
        sol = diffrax.diffeqsolve(
            system, diffrax.Euler(), 0, 10, 0.01, jnp.zeros((2,2)), max_steps=16**4
        )
        return sol.ys[0]

    def evaluate_LQG(self, x0: Sequence[float], ts: Sequence[float], target: float, process_noise_key, obs_noise_key, params: Tuple):
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

        #apply process noise only on x
        def diffusion(t, variables, args):
            x = variables[:self.latent_size]
            return jnp.concatenate([env.diffusion(t,x,args),jnp.zeros((self.latent_size,self.latent_size)),jnp.zeros((self.latent_size**2,self.latent_size))])
        
        solver = diffrax.EulerHeun()
        dt0 = 0.005
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
        u = jax.vmap(lambda m, l: -l@(m-x_star) + u_star, in_axes=[0,None])(mu, L) #Map states to control
        _, y = jax.lax.scan(env.f_obs, obs_noise_key, (ts, x)) #Map states to observations

        costs = env.fitness_function(x, u, target)

        return x, y, u, costs

    def __call__(self, data):
        x, y, u, costs = jax.vmap(self.evaluate_LQG, in_axes=[0, None, 0, 0, 0, 0])(*data)
        
        return x, y, u, jnp.mean(costs)