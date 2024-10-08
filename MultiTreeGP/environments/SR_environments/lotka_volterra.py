import jax
import jax.numpy as jnp
import jax.random as jrandom
from MultiTreeGP.environments.SR_environments.time_series_environment_base import EnvironmentBase

class LotkaVolterra(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs=2):
        n_var = 2
        super().__init__(process_noise, obs_noise, n_var, n_obs)

        self.init_mu = jnp.array([10, 10])
        self.init_sd = 2

        self.alpha = 1.1
        self.beta = 0.4
        self.delta = 0.1
        self.gamma = 0.4
        self.V = self.process_noise * jnp.eye(self.n_var)
        self.W = self.obs_noise * jnp.eye(self.n_obs)[:self.n_obs]
        self.C = jnp.eye(self.n_var)[:self.n_obs]

    def sample_init_states(self, batch_size, key):
        return self.init_mu + self.init_sd*jrandom.normal(key, shape=(batch_size,2))
        # return jrandom.uniform(key, shape = (batch_size,2), minval=5, maxval=15)
    
    def sample_init_state2(self, ys, batch_size, key):
        return ys[jrandom.choice(key, jnp.arange(ys.shape[0]), shape=(batch_size,), replace=False)]
    
    def drift(self, t, state, args):
        return jnp.array([self.alpha * state[0] - self.beta * state[0] * state[1], self.delta * state[0] * state[1] - self.gamma * state[1]])

    def diffusion(self, t, state, args):
        return self.V

    def terminate_event(self, state, **kwargs):
        return False