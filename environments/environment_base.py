import jax
import jax.numpy as jnp
import jax.random as jrandom

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

class Environment:
    def __init__(self, key, sigma, obs_noise, n_obs, n_var, n_control, mu0, P0):
        self.n_obs = n_obs
        self.sigma = sigma
        self.obs_noise = obs_noise
        self.n_var = n_var
        self.n_control = n_control
        
        self.key = key
        self.mu0 = mu0
        self.P0 = P0

    def initialize_parameters(self, params):
        pass

    def f_obs(self, t, x):
        key = jrandom.fold_in(self.key, force_bitcast_convert_type(t))
        return self.C@x + jrandom.normal(key, shape=(self.n_obs,))*self.W
    
    def drift(self, t, state, args):
        pass

    def diffusion(self, t, state, args):
        pass

    def fitness_function(self, states, controls, targets):
        pass