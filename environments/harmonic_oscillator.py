import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax
from environments.environment_base import EnvironmentBase

class HarmonicOscillator(EnvironmentBase):
    def __init__(self, sigma, obs_noise, n_obs = None):
        self.n_dim = 1
        self.n_var = 2
        self.n_control = 1
        self.n_targets = 1
        self.mu0 = jnp.zeros(self.n_var)
        self.P0 = jnp.eye(self.n_var)
        super().__init__(sigma, obs_noise, self.n_var, self.n_control, self.n_dim, n_obs)

        self.q = self.r = 0.5
        self.Q = jnp.array([[self.q,0],[0,0]])
        self.R = jnp.array([[self.r]])

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = self.mu0 + jrandom.normal(init_key, shape=(batch_size, self.n_var))@self.P0
        targets = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=-3, maxval=3)
        return  x0, targets
    
    def sample_params(self, batch_size, mode, ts, key):
        omega_key, zeta_key, args_key = jrandom.split(key, 3)
        if mode == "Constant":
            omegas = jnp.ones((batch_size, ts.shape[0]))
            zetas = jnp.zeros((batch_size, ts.shape[0]))
            # omegas = jnp.ones((batch_size))
            # zetas = jnp.zeros((batch_size))
        elif mode == "Different":
            omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.5, maxval=1.5)[:,None] * jnp.ones((batch_size,ts.shape[0]))
            zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.0)[:,None] * jnp.ones((batch_size,ts.shape[0]))
        elif mode == "Switch":
            switch_times = jrandom.randint(args_key, shape=(batch_size,), minval=int(ts.shape[0]/4), maxval=int(3*ts.shape[0]/4))
            omegas = jnp.zeros((batch_size, ts.shape[0]))
            zetas = jnp.zeros((batch_size, ts.shape[0]))
            for i in range(batch_size):
                _key11, _key12 = jrandom.split(jrandom.fold_in(omega_key, i))
                omegas = omegas.at[i,:switch_times[i]].set(jrandom.uniform(_key11, shape=(), minval=0.5, maxval=1.5))
                omegas = omegas.at[i,switch_times[i]:].set(jrandom.uniform(_key12, shape=(), minval=0.5, maxval=1.5))

                _key21, _key22 = jrandom.split(jrandom.fold_in(zeta_key, i))
                zetas = zetas.at[i,:switch_times[i]].set(jrandom.uniform(_key21, shape=(), minval=0., maxval=1.))
                zetas = zetas.at[i,switch_times[i]:].set(jrandom.uniform(_key22, shape=(), minval=0., maxval=1.))

        elif mode == "Decay":
            decay_factors = jrandom.uniform(args_key, shape=(batch_size,2), minval=0.98, maxval=1.02)
            init_omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.5, maxval=1.5)
            init_zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.0)
            omegas = jax.vmap(lambda o, d, t: o*(d**t), in_axes=[0, 0, None])(init_omegas, decay_factors[:,0], ts)
            zetas = jax.vmap(lambda z, d, t: z*(d**t), in_axes=[0, 0, None])(init_zetas, decay_factors[:,1], ts)

        return omegas, zetas

    def initialize_parameters(self, params, ts):
        omega, zeta = params

        A = jax.vmap(lambda o, z: jnp.array([[0,1],[-(o),-z]]))(omega, zeta)
        self.A = diffrax.LinearInterpolation(ts, A)

        self.b = jnp.array([[0.0,1.0]]).T
        self.G = jnp.array([[0,0],[0,1]])
        self.V = self.sigma*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)

    def drift(self, t, state, args):
        # print(self.A.shape, state.shape, self.b.shape, args.shape)
        return self.A.evaluate(t)@state + self.b@args
    
    def diffusion(self, t, state, args):
        return self.V
    
    def fitness_function(self, state, control, target, ts):
        x_d = jnp.array([jnp.squeeze(target), 0])

        u_d = jax.vmap(lambda t: -jnp.linalg.pinv(self.b)@self.A.evaluate(t)@x_d)(ts)
        costs = jax.vmap(lambda _state, _u, _u_d: (_state-x_d).T@self.Q@(_state-x_d) + (_u-_u_d)@self.R@(_u-_u_d))(state,control,u_d)
        return jnp.sum(costs)
    
    def terminate_event(self, state, **kwargs):
        return jnp.any(jnp.isnan(state.y))# | jnp.any(jnp.isinf(state.y))

class HarmonicOscillator2D(EnvironmentBase):
    def __init__(self, sigma, obs_noise, n_obs, n_dim):
        self.n_dim = n_dim
        self.n_var = 2
        self.n_control = n_dim
        self.n_targets = n_dim
        self.mu0 = jnp.zeros(self.n_var*self.n_dim)
        self.P0 = jnp.eye(self.n_var*self.n_dim)
        super().__init__(sigma, obs_noise, n_obs, self.n_var, self.n_control, self.n_dim)

        self.q = self.r = 0.5
        self.Q = self.block_diagonal(jnp.array([[self.q,0],[0,0]]))
        self.R = self.block_diagonal(jnp.array([[self.r]]))

    def sample_init_states(self, batch_size, key):
        return self.mu0 + jrandom.normal(key, shape=(batch_size, self.n_var*self.n_dim))@self.P0
    
    def sample_params(self, batch_size, mode, ts, key):
        return jnp.zeros(batch_size)

    def block_diagonal(self, block):
        dim1, dim2 = block.shape
        result = jnp.zeros((self.n_dim*dim1, self.n_dim*dim2))
        for i in range(self.n_dim):
            result = result.at[i*dim1:(i+1)*dim1,i*dim2:(i+1)*dim2].set(block)
        return result

    def initialize_parameters(self, params, ts):
        _ = params

        self.A = self.block_diagonal(jnp.array([[0,1],[-1,0]]))

        self.b = self.block_diagonal(jnp.array([[0.0,1.0]]).T)
        self.G = self.block_diagonal(jnp.array([[0,0],[0,1]]))
        self.V = self.sigma*self.G

        indices = jnp.array([jnp.arange(i*self.n_var, (i+1)*self.n_var)[:self.n_obs] for i in range(self.n_dim)])
        self.C = jnp.eye(self.n_var*self.n_dim)[jnp.ravel(indices)]
        self.W = self.obs_noise*jnp.eye(self.n_obs*self.n_dim)

    def drift(self, t, state, args):
        # print(self.A.shape, state.shape, self.b.shape, args.shape)
        return self.A@state + self.b@args
    
    def diffusion(self, t, state, args):
        return self.V
    
    def fitness_function(self, state, control, target, ts):
        x_d = jnp.zeros((self.n_var*self.n_dim))
        for i in range(self.n_dim):
            x_d = x_d.at[i*self.n_var].set(target[i])

        u_d = -jnp.linalg.pinv(self.b)@self.A@x_d
        costs = jax.vmap(lambda _state, _u: (_state-x_d).T@self.Q@(_state-x_d) + (_u-u_d)@self.R@(_u-u_d))(state,control)
        return jnp.sum(costs)
    
    def terminate_event(self, state, **kwargs):
        return jnp.any(jnp.isnan(state.y))# | jnp.any(jnp.isinf(state.y))