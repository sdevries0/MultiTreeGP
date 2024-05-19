import jax
import jax.numpy as jnp
import jax.random as jrandom
from environments.control_environments.control_environment_base import EnvironmentBase

class CartPole(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs = 4):
        self.n_var = 4
        self.n_control = 1
        self.n_targets = 0
        self.n_dim = 1
        self.init_bounds = jnp.array([0.05,0.05,0.05,0.05])
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control, self.n_dim, n_obs)

        self.Q = jnp.array(0)
        self.R = jnp.array([[0.0]])

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval= -self.init_bounds, maxval= self.init_bounds)
        targets = jnp.zeros((batch_size, self.n_targets))
        return x0, targets
    
    def sample_params(self, batch_size, mode, ts, key):
        #g, mass, length
        return jnp.zeros((batch_size))

    def initialize_parameters(self, params, ts):
        _ = params
        self.g = 9.81
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.cart_mass = 1
        
        self.G = jnp.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
        self.V = self.process_noise*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)

    def drift(self, t, state, args):
        control = jnp.squeeze(args)
        control = jnp.clip(control, -1, 1)
        x, theta, x_dot, theta_dot = state

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)
        
        theta_acc = (self.g * sin_theta - cos_theta * (
            control + self.pole_mass * self.pole_length * theta_dot**2 * sin_theta
            ) / (self.cart_mass + self.pole_mass)) / (
                self.pole_length * (4/3 - (self.pole_mass * cos_theta**2) / (self.cart_mass + self.pole_mass)))

        x_acc = (control + self.pole_mass * self.pole_length * (theta_dot**2 * sin_theta - theta_acc * cos_theta)) / (self.cart_mass + self.pole_mass)

        # x_acc = control
        # theta_acc = (self.g/self.pole_length) * jnp.sin(theta)

        return jnp.array([
            x_dot,
            theta_dot,
            x_acc,
            theta_acc
        ])
    
    def diffusion(self, t, x, args):
        return self.V
    
    def fitness_function(self, state, control, target, ts):
        invalid_points = jax.vmap(lambda _x, _u: jnp.any(jnp.isinf(_x)) + jnp.isnan(_u))(state, control[:,0])
        punishment = jnp.ones_like(invalid_points)
        # costs_control = jax.vmap(lambda _state, _u: _u@self.R@_u)(state, u)

        costs = jnp.where(invalid_points, punishment, jnp.zeros_like(punishment))

        return jnp.sum(costs)
    
    def terminate_event(self, state, **kwargs):
        return (jnp.abs(state.y[1])>0.2) | (jnp.abs(state.y[0])>4.8) | jnp.any(jnp.isnan(state.y)) | jnp.any(jnp.isinf(state.y))