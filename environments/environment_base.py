import jax
import jax.numpy as jnp
import jax.random as jrandom
import abc

_itemsize_kind_type = {
    (1, "i"): jnp.int8,
    (2, "i"): jnp.int16,
    (4, "i"): jnp.int32,
    (8, "i"): jnp.int64,
    (2, "f"): jnp.float16,
    (4, "f"): jnp.float32,
    (8, "f"): jnp.float64,
}

def force_bitcast_convert_type(val, new_type=jnp.int32):
    val = jnp.asarray(val)
    intermediate_type = _itemsize_kind_type[new_type.dtype.itemsize, val.dtype.kind]
    val = val.astype(intermediate_type)
    return jax.lax.bitcast_convert_type(val, new_type)

class EnvironmentBase(abc.ABC):
    def __init__(self, sigma, obs_noise, n_obs, n_var, n_control, n_dim):
        self.sigma = sigma
        self.obs_noise = obs_noise
        self.n_var = n_var
        self.n_control = n_control
        self.n_dim = n_dim
        if n_obs:
            self.n_obs = n_obs
        else:
            self.n_obs = self.n_var

    @abc.abstractmethod
    def initialize_parameters(self, params, ts):
        return

    @abc.abstractmethod
    def sample_init_states(self, batch_size, key):
        return

    @abc.abstractmethod
    def sample_params(self, batch_size, mode, ts, key):
        return

    def f_obs(self, key, t_x):
        t, x = t_x
        new_key = jrandom.fold_in(key, force_bitcast_convert_type(t))
        out = self.C@x + jrandom.normal(new_key, shape=(self.n_obs*self.n_dim,))@self.W
        return key, out
    
    @abc.abstractmethod
    def drift(self, t, state, args):
        return

    @abc.abstractmethod
    def diffusion(self, t, state, args):
        return

    @abc.abstractmethod
    def fitness_function(self, state, control, target, ts):
        return 100

    def terminate_event(self, state, **kwargs):
        return False