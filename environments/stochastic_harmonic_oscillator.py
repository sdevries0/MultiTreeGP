import jax
import jax.numpy as jnp
import diffrax
from environments.environment_base import Environment

class StochasticHarmonicOscillator(Environment):
    def __init__(self, key, sigma, obs_noise, n_obs):
        self.n_var = 2
        self.n_control = 1
        mu0 = jnp.zeros(2)
        P0 = jnp.eye(2)
        super().__init__(key, sigma, obs_noise, n_obs, self.n_var, self.n_control, mu0, P0)

        self.q = self.r = 0.5
        self.Q = jnp.array([[self.q,0],[0,0]])
        self.R = jnp.array([[self.r]])
        
    def initialize_parameters(self, params):
        omega, zeta = params
        self.A = jnp.array([[0,1],[-(omega**2),-zeta]])
        self.b = jnp.array([[0.0,1.0]]).T
        self.G = jnp.array([[0,0],[0,1]])
        self.V = self.sigma*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.ones(self.n_obs)

    def compute_riccati(self):
        def riccati(t,S,args):
            return self.A.T@S + S@self.A - S@self.b@(1/self.R)@self.b.T@S + self.Q

        sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(riccati), diffrax.Euler(), 0,10, 0.01, self.Q, max_steps=16**4
                )
        L = 1/self.R@self.b.T@sol.ys[0]
        return L[0]

    def drift(self, t, state, args):
        return self.A@state + self.b@args
    
    def diffusion(self, t, state, args):
        return self.V
    
    def fitness_function(self, state, u, target):
        x_d = jnp.array([target,0])
        u_d = -jnp.linalg.pinv(self.b)@self.A@x_d
        costs = jax.vmap(lambda _state, _u: (_state-x_d).T@self.Q@(_state-x_d) + (_u-u_d)@self.R@(_u-u_d))(state,u)
        return jnp.cumsum(costs)