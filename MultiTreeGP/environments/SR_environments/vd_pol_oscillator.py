import jax
import jax.numpy as jnp
import jax.random as jrandom
from MultiTreeGP.environments.SR_environments.time_series_environment_base import EnvironmentBase

class VanDerPolOscillator(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs=2):
        n_var = 2
        super().__init__(process_noise, obs_noise, n_var, n_obs)

        self.init_mu = jnp.array([0,0])
        self.init_sd = jnp.array([1.0,1.0])

        self.mu = 1
        self.V = self.process_noise * jnp.eye(self.n_var)
        self.W = self.obs_noise * jnp.eye(self.n_obs)[:self.n_obs]
        self.C = jnp.eye(self.n_var)[:self.n_obs]

    def sample_init_states(self, batch_size, key):
        return self.init_mu + self.init_sd*jrandom.normal(key, shape=(batch_size,2))
    
    def drift(self, t, state, args):
        return jnp.array([state[1], self.mu*(1-state[0]**2)*state[1] - state[0]])

    def diffusion(self, t, state, args):
        return self.V

    def terminate_event(self, state, **kwargs):
        return False