import jax
import jax.numpy as jnp

from environments.environment_base import Environment

class CartPole(Environment):
    def __init__(self, key, sigma, obs_noise, n_obs):
        self.n_var = 4
        self.n_control = 1
        mu0 = jnp.array([0,0,0,0])
        P0 = jnp.eye(4)*jnp.array([1,0.1,0.1,0.2])
        super().__init__(key, sigma, obs_noise, n_obs, self.n_var, self.n_control, mu0, P0)

        self.threshold_theta = 30 * 2 * jnp.pi / 360
        self.threshold_x = 5

    def initialize_parameters(self, params):
        _ = params
        self.g = 9.81
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.cart_mass = 1
        
        self.G = jnp.array([[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
        self.V = self.sigma*self.G

        self.C = jnp.eye(self.n_var)[:,:self.n_obs]
        self.W = self.obs_noise*jnp.ones(self.n_obs)

    def drift(self, t, state, args):
        control = jnp.squeeze(args)
        # control = control / (jnp.abs(control)+1e-5)
        x, x_dot, theta, theta_dot = state
        
        theta_acc = (self.g * jnp.sin(theta) + jnp.cos(theta) * (
            -control - self.pole_mass * self.pole_length * theta_dot**2 * jnp.sin(theta)
            ) / (self.cart_mass + self.pole_mass)) / (
                self.pole_length * (4/3 - self.pole_mass * jnp.cos(theta)**2 / (self.cart_mass + self.pole_mass)))
        x_acc = (control + self.pole_mass * self.pole_length * (theta_dot**2 * jnp.sin(theta) - theta_acc * jnp.cos(theta))) / (self.cart_mass + self.pole_mass)

        return jnp.array([
            x_dot,
            x_acc,
            theta_dot,
            theta_acc
        ])
    
    def diffusion(self, t, x, args):
        return self.V
    
    def fitness_function(self, state, u, target):
        u = jnp.squeeze(u)
        x_d = jnp.array([target,0])
        # angle_rewards = jnp.cos(state[:,2])>self.threshold_theta
        angle_rewards = jnp.cos(state[:,2])>self.threshold_theta
        position_rewards = jnp.abs(state[:,0])<self.threshold_x

        costs = jnp.cumsum(1.0-angle_rewards*position_rewards)# - 1e5*jnp.square(u))
        return costs