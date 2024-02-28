import jax
import jax.numpy as jnp
import jax.random as jrandom
from environments.environment_base import EnvironmentBase

class Acrobot(EnvironmentBase):
    def __init__(self, sigma, obs_noise, n_obs = None):
        self.n_var = 4
        self.n_control = 1
        self.n_targets = 0
        self.n_dim = 1
        self.init_bounds = jnp.array([0.1,0.1,0.1,0.1])
        self.default_obs = 6
        self.obs_bounds = (-jnp.array([1, 1, 1, 1, jnp.inf, jnp.inf]), jnp.array([1, 1, 1, 1, jnp.inf, jnp.inf]))
        super().__init__(sigma, obs_noise, self.n_var, self.n_control, self.n_dim, self.obs_bounds, n_obs if n_obs else self.default_obs)

        self.Q = jnp.array(0.0)
        self.R = jnp.array([[0.0]])

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval= -self.init_bounds, maxval= self.init_bounds)
        targets = jnp.zeros((batch_size, self.n_targets))
        return x0, targets
    
    def sample_params(self, batch_size, mode, ts, key):
        #g, l1, l2, m1, m2, lc1, lc2
        return jnp.zeros(batch_size)
    
    def f_obs(self, key, t_x):
        t, x = t_x
        x_ = jnp.array([jnp.cos(x[0]), jnp.sin(x[0]), jnp.cos(x[1]), jnp.sin(x[1]), x[2], x[3]])
        new_key, out = super().f_obs(key, (t, x_))
        return key, out

    def initialize_parameters(self, params, ts):
        _ = params
        self.l1 = 1.0  # [m]
        self.l2 = 1.0  # [m]
        self.m1 = 1.0  #: [kg] mass of link 1
        self.m2 = 1.0  #: [kg] mass of link 2
        self.lc1 = 0.5  #: [m] position of the center of mass of link 1
        self.lc2 = 0.5  #: [m] position of the center of mass of link 2
        self.moi1 = self.moi2 = 1.0
        self.g = 9.81

        self.G = jnp.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
        self.V = self.sigma*self.G

        self.C = jnp.eye(6)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)*(jnp.array([0.5,0.5,0.5,0.5,1,1])[:self.n_obs])
    
    def drift(self, t, state, args):
        control = jnp.squeeze(args)
        control = 2*jnp.tanh(control)
        theta1, theta2, theta1_dot, theta2_dot = state

        d1 = self.m1 * self.lc1**2 + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * jnp.cos(theta2)) + self.moi1 + self.moi2
        d2 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * jnp.cos(theta2)) + self.moi2

        phi2 = self.m2 * self.lc2 * self.g * jnp.cos(theta1 + theta2 - jnp.pi/2)
        phi1 = -self.m2 * self.l1 * self.lc2 * theta2_dot**2 * jnp.sin(theta2) - 2 * self.m2 * self.l1 * self.lc2 * theta1_dot * theta2_dot * jnp.sin(theta1) \
                    + (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * jnp.cos(theta1 - jnp.pi/2) + phi2
        
        theta2_acc = (control + d2/d1 * phi1 - self.m2 * self.l1 * self.lc2 * theta1_dot**2 * jnp.sin(theta2) - phi2) \
                    / (self.m2 * self.lc2**2 + self.moi2 - d2**2 / d1)
        theta1_acc = -(d2 * theta2_acc + phi1)/d1

        return jnp.array([
            theta1_dot,
            theta2_dot,
            theta1_acc,
            theta2_acc
        ])

    def diffusion(self, t, state, args):
        return self.V

    def fitness_function(self, state, control, target, ts):
        # w1 = 1
        # w2 = 1

        # invalid_points = jax.vmap(lambda _x, _u: jnp.any(jnp.isinf(_x)) + jnp.isnan(_u))(state, control[:,0])
        reached_threshold = jax.vmap(lambda theta1, theta2: -jnp.cos(theta1) - jnp.cos(theta1 + theta2) > 1.5)(state[:,0], state[:,1])
        # angle_distance = jax.vmap(lambda theta1, theta2: (2 + jnp.cos(theta1) + jnp.cos(theta1 + theta2))/4)(state[:,0], state[:,1])
        # costs = jnp.where(invalid_points, jnp.ones_like(ts), angle_distance)

        # return w1* jnp.sum(1-reached_threshold) + w2 * jnp.sum(costs)
    
        # # slow_angle = jax.vmap(lambda theta_dot1, theta_dot2: jnp.abs(theta_dot1)+jnp.abs(theta_dot2) < 5)(state[:,2], state[:,3])
        

        # control_cost = jax.vmap(lambda _state, _u: _u@self.R@_u)(state, control)
        # costs = jnp.where((ts/(ts[1]-ts[0]))>first_success, jnp.zeros_like(control_cost), control_cost)

        first_success = jnp.argmax(reached_threshold)
        return first_success + (first_success == 0) * ts.shape[0]  # + jnp.sum(costs)

        # invalid_points = jax.vmap(lambda _x, _u: jnp.any(jnp.isinf(_x)) + jnp.isnan(_u))(state, control[:,0])
        # angle_distance = jax.vmap(lambda theta1, theta2: ((2 + jnp.cos(theta1) + jnp.cos(theta1 + theta2))/4)**2)(state[:,0], state[:,1])
        # angle_costs = jnp.where(invalid_points, jnp.ones_like(ts), angle_distance)
        # reached_threshold = jax.vmap(lambda theta1, theta2: -jnp.cos(theta1) - jnp.cos(theta1 + theta2) > 0.0)(state[:,0], state[:,1])
        # first_success = jnp.argmax(reached_threshold) + (jnp.sum(reached_threshold)==0) * ts.shape[0]
        # costs = jnp.where((ts/(ts[1]-ts[0]))<first_success, jnp.ones_like(angle_costs), angle_costs)

        # return jnp.sum(costs)

    def terminate_event(self, state, **kwargs):
        return (jnp.abs(state.y[2])>(4*jnp.pi)) | (jnp.abs(state.y[3])>(9*jnp.pi)) | jnp.any(jnp.isnan(state.y)) | jnp.any(jnp.isinf(state.y))