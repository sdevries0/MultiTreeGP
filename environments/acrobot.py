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
        self.default_obs = 4
        super().__init__(sigma, obs_noise, self.n_var, self.n_control, self.n_dim, n_obs if n_obs else self.default_obs)

        self.R = jnp.array([[0.01]])

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval= -self.init_bounds, maxval= self.init_bounds)
        targets = jnp.zeros((batch_size, self.n_targets))
        return x0, targets
    
    def sample_params(self, batch_size, mode, ts, key):
        l1_key, l2_key, m1_key, m2_key, args_key = jrandom.split(key, 5)
        if mode == "Constant":
            l1 = l2 = m1 = m2 = jnp.ones((batch_size))
        elif mode == "Different":
            l1 = jrandom.uniform(l1_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            l2 = jrandom.uniform(l2_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            m1 = jrandom.uniform(m1_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            m2 = jrandom.uniform(m2_key, shape=(batch_size,), minval=0.75, maxval=1.25)
        elif mode == "Switch":
            switch_times = jrandom.randint(args_key, shape=(batch_size,), minval=int(ts.shape[0]/4), maxval=int(3*ts.shape[0]/4))
            l1 = jnp.zeros((batch_size, ts.shape[0]))
            l2 = jnp.zeros((batch_size, ts.shape[0]))
            m1 = jnp.zeros((batch_size, ts.shape[0]))
            m2 = jnp.zeros((batch_size, ts.shape[0]))
            for i in range(batch_size):
                _key11, _key12 = jrandom.split(jrandom.fold_in(l1_key, i))
                l1 = l1.at[i,:switch_times[i]].set(jrandom.uniform(_key11, shape=(), minval=0.75, maxval=1.25))
                l1 = l1.at[i,switch_times[i]:].set(jrandom.uniform(_key12, shape=(), minval=0.75, maxval=1.25))

                _key21, _key22 = jrandom.split(jrandom.fold_in(l2_key, i))
                l2 = l2.at[i,:switch_times[i]].set(jrandom.uniform(_key21, shape=(), minval=0.75, maxval=1.25))
                l2 = l2.at[i,switch_times[i]:].set(jrandom.uniform(_key22, shape=(), minval=0.75, maxval=1.25))

                _key31, _key32 = jrandom.split(jrandom.fold_in(m1_key, i))
                m1 = m1.at[i,:switch_times[i]].set(jrandom.uniform(_key31, shape=(), minval=0.75, maxval=1.25))
                m1 = m1.at[i,switch_times[i]:].set(jrandom.uniform(_key32, shape=(), minval=0.75, maxval=1.25))

                _key41, _key42 = jrandom.split(jrandom.fold_in(m2_key, i))
                m2 = m2.at[i,:switch_times[i]].set(jrandom.uniform(_key41, shape=(), minval=0.75, maxval=1.25))
                m2 = m2.at[i,switch_times[i]:].set(jrandom.uniform(_key42, shape=(), minval=0.75, maxval=1.25))

        elif mode == "Decay":
            decay_factors = jrandom.uniform(args_key, shape=(batch_size,2), minval=0.98, maxval=1.02)
            init_l1 = jrandom.uniform(l1_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            init_l2 = jrandom.uniform(l2_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            init_m1 = jrandom.uniform(m1_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            init_m2 = jrandom.uniform(m2_key, shape=(batch_size,), minval=0.75, maxval=1.25)

            l1 = jax.vmap(lambda l, d, t: l*(d**t), in_axes=[0, 0, None])(init_l1, decay_factors[:,0], ts)
            l2 = jax.vmap(lambda l, d, t: l*(d**t), in_axes=[0, 0, None])(init_l2, decay_factors[:,1], ts)
            m1 = jax.vmap(lambda m, d, t: m*(d**t), in_axes=[0, 0, None])(init_m1, decay_factors[:,0], ts)
            m2 = jax.vmap(lambda m, d, t: m*(d**t), in_axes=[0, 0, None])(init_m2, decay_factors[:,1], ts)

        return l1, l2, m1, m2
    
    def f_obs(self, key, t_x):
        _, out = super().f_obs(key, t_x)
        out = jnp.array([(out[0]+jnp.pi)%(2*jnp.pi)-jnp.pi, (out[1]+jnp.pi)%(2*jnp.pi)-jnp.pi, out[2], out[3]])[:self.n_obs]
        return key, out

    def initialize_parameters(self, params, ts):
        l1, l2, m1, m2 = params
        self.l1 = l1  # [m]
        self.l2 = l2  # [m]
        self.m1 = m1  #: [kg] mass of link 1
        self.m2 = m2  #: [kg] mass of link 2
        self.lc1 = 0.5*self.l1  #: [m] position of the center of mass of link 1
        self.lc2 = 0.5*self.l2  #: [m] position of the center of mass of link 2
        self.moi1 = self.moi2 = 1.0
        self.g = 9.81

        self.G = jnp.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
        self.V = self.sigma*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)*(jnp.array([1,1,1,1])[:self.n_obs])
    
    def drift(self, t, state, args):
        control = jnp.squeeze(args)
        control = jnp.clip(control, -1, 1)
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
        reached_threshold = jax.vmap(lambda theta1, theta2: -jnp.cos(theta1) - jnp.cos(theta1 + theta2) > 1.5)(state[:,0], state[:,1])
        first_success = jnp.argmax(reached_threshold)

        control_cost = jax.vmap(lambda _state, _u: _u@self.R@_u)(state, control)
        costs = jnp.where((ts/(ts[1]-ts[0]))>first_success, jnp.zeros_like(control_cost), control_cost)

        return first_success + (first_success == 0) * ts.shape[0] + jnp.sum(costs)

    def terminate_event(self, state, **kwargs):
        return (jnp.abs(state.y[2])>(8*jnp.pi)) | (jnp.abs(state.y[3])>(18*jnp.pi)) | jnp.any(jnp.isnan(state.y)) | jnp.any(jnp.isinf(state.y))

class Acrobot2(EnvironmentBase):
    def __init__(self, sigma, obs_noise, n_obs = None):
        self.n_var = 4
        self.n_control = 2
        self.n_targets = 0
        self.n_dim = 1
        self.init_bounds = jnp.array([0.1,0.1,0.1,0.1])
        self.default_obs = 4
        super().__init__(sigma, obs_noise, self.n_var, self.n_control, self.n_dim, n_obs if n_obs else self.default_obs)

        self.R = jnp.array(0.01)*jnp.ones((self.n_control, self.n_control))

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval= -self.init_bounds, maxval= self.init_bounds)
        targets = jnp.zeros((batch_size, self.n_targets))
        return x0, targets
    
    def sample_params(self, batch_size, mode, ts, key):
        l1_key, l2_key, m1_key, m2_key, args_key = jrandom.split(key, 5)
        if mode == "Constant":
            l1 = l2 = m1 = m2 = jnp.ones((batch_size))
        elif mode == "Different":
            l1 = jrandom.uniform(l1_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            l2 = jrandom.uniform(l2_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            m1 = jrandom.uniform(m1_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            m2 = jrandom.uniform(m2_key, shape=(batch_size,), minval=0.75, maxval=1.25)
        elif mode == "Switch":
            switch_times = jrandom.randint(args_key, shape=(batch_size,), minval=int(ts.shape[0]/4), maxval=int(3*ts.shape[0]/4))
            l1 = jnp.zeros((batch_size, ts.shape[0]))
            l2 = jnp.zeros((batch_size, ts.shape[0]))
            m1 = jnp.zeros((batch_size, ts.shape[0]))
            m2 = jnp.zeros((batch_size, ts.shape[0]))
            for i in range(batch_size):
                _key11, _key12 = jrandom.split(jrandom.fold_in(l1_key, i))
                l1 = l1.at[i,:switch_times[i]].set(jrandom.uniform(_key11, shape=(), minval=0.75, maxval=1.25))
                l1 = l1.at[i,switch_times[i]:].set(jrandom.uniform(_key12, shape=(), minval=0.75, maxval=1.25))

                _key21, _key22 = jrandom.split(jrandom.fold_in(l2_key, i))
                l2 = l2.at[i,:switch_times[i]].set(jrandom.uniform(_key21, shape=(), minval=0.75, maxval=1.25))
                l2 = l2.at[i,switch_times[i]:].set(jrandom.uniform(_key22, shape=(), minval=0.75, maxval=1.25))

                _key31, _key32 = jrandom.split(jrandom.fold_in(m1_key, i))
                m1 = m1.at[i,:switch_times[i]].set(jrandom.uniform(_key31, shape=(), minval=0.75, maxval=1.25))
                m1 = m1.at[i,switch_times[i]:].set(jrandom.uniform(_key32, shape=(), minval=0.75, maxval=1.25))

                _key41, _key42 = jrandom.split(jrandom.fold_in(m2_key, i))
                m2 = m2.at[i,:switch_times[i]].set(jrandom.uniform(_key41, shape=(), minval=0.75, maxval=1.25))
                m2 = m2.at[i,switch_times[i]:].set(jrandom.uniform(_key42, shape=(), minval=0.75, maxval=1.25))

        elif mode == "Decay":
            decay_factors = jrandom.uniform(args_key, shape=(batch_size,2), minval=0.98, maxval=1.02)
            init_l1 = jrandom.uniform(l1_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            init_l2 = jrandom.uniform(l2_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            init_m1 = jrandom.uniform(m1_key, shape=(batch_size,), minval=0.75, maxval=1.25)
            init_m2 = jrandom.uniform(m2_key, shape=(batch_size,), minval=0.75, maxval=1.25)

            l1 = jax.vmap(lambda l, d, t: l*(d**t), in_axes=[0, 0, None])(init_l1, decay_factors[:,0], ts)
            l2 = jax.vmap(lambda l, d, t: l*(d**t), in_axes=[0, 0, None])(init_l2, decay_factors[:,1], ts)
            m1 = jax.vmap(lambda m, d, t: m*(d**t), in_axes=[0, 0, None])(init_m1, decay_factors[:,0], ts)
            m2 = jax.vmap(lambda m, d, t: m*(d**t), in_axes=[0, 0, None])(init_m2, decay_factors[:,1], ts)

        return l1, l2, m1, m2
    
    def f_obs(self, key, t_x):
        _, out = super().f_obs(key, t_x)
        out = jnp.array([(out[0]+jnp.pi)%(2*jnp.pi)-jnp.pi, (out[1]+jnp.pi)%(2*jnp.pi)-jnp.pi, out[2], out[3]])[:self.n_obs]
        return key, out

    def initialize_parameters(self, params, ts):
        l1, l2, m1, m2 = params
        self.l1 = l1  # [m]
        self.l2 = l2  # [m]
        self.m1 = m1  #: [kg] mass of link 1
        self.m2 = m2  #: [kg] mass of link 2
        self.lc1 = 0.5*self.l1  #: [m] position of the center of mass of link 1
        self.lc2 = 0.5*self.l2  #: [m] position of the center of mass of link 2
        self.moi1 = self.moi2 = 1.0
        self.g = 9.81

        self.G = jnp.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
        self.V = self.sigma*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)*(jnp.array([1,1,1,1])[:self.n_obs])
    
    def drift(self, t, state, args):
        control = jnp.squeeze(args)
        control = jnp.clip(control, -1, 1)
        c1, c2 = control
        theta1, theta2, theta1_dot, theta2_dot = state

        d1 = self.m1 * self.lc1**2 + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * jnp.cos(theta2)) + self.moi1 + self.moi2
        d2 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * jnp.cos(theta2)) + self.moi2

        phi2 = self.m2 * self.lc2 * self.g * jnp.cos(theta1 + theta2 - jnp.pi/2)
        phi1 = -self.m2 * self.l1 * self.lc2 * theta2_dot**2 * jnp.sin(theta2) - 2 * self.m2 * self.l1 * self.lc2 * theta1_dot * theta2_dot * jnp.sin(theta1) \
                    + (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * jnp.cos(theta1 - jnp.pi/2) + phi2
        
        theta2_acc = (c1 + d2/d1 * phi1 - self.m2 * self.l1 * self.lc2 * theta1_dot**2 * jnp.sin(theta2) - phi2) \
                    / (self.m2 * self.lc2**2 + self.moi2 - d2**2 / d1)
        theta1_acc = (c2 - d2 * theta2_acc + phi1)/d1

        return jnp.array([
            theta1_dot,
            theta2_dot,
            theta1_acc,
            theta2_acc
        ])

    def diffusion(self, t, state, args):
        return self.V

    def fitness_function(self, state, control, target, ts):
        reached_threshold = jax.vmap(lambda theta1, theta2: -jnp.cos(theta1) - jnp.cos(theta1 + theta2) > 1.5)(state[:,0], state[:,1])
        first_success = jnp.argmax(reached_threshold)

        control_cost = jax.vmap(lambda _state, _u: _u@self.R@_u)(state, control)
        costs = jnp.where((ts/(ts[1]-ts[0]))>first_success, jnp.zeros_like(control_cost), control_cost)

        return first_success + (first_success == 0) * ts.shape[0] + jnp.sum(costs)

    def terminate_event(self, state, **kwargs):
        return (jnp.abs(state.y[2])>(8*jnp.pi)) | (jnp.abs(state.y[3])>(18*jnp.pi)) | jnp.any(jnp.isnan(state.y)) | jnp.any(jnp.isinf(state.y))