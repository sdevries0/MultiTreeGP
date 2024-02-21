from environments.environment_base import EnvironmentBase
import jax.numpy as jnp
import jax
import jax.random as jrandom

class StirredTankReactor(EnvironmentBase):
    def __init__(self, sigma, obs_noise, n_obs = None):
        self.sigma = sigma
        self.obs_noise = obs_noise
        self.n_var = 3
        self.n_control = 1
        self.n_dim = 1
        self.n_targets = 1
        self.init_lower_bounds = jnp.array([200, 200, 0.0])
        self.init_upper_bounds = jnp.array([300, 400, 1.0])
        super().__init__(sigma, obs_noise, self.n_var, self.n_control, self.n_dim, n_obs)

        self.Q = jnp.array([[0,0,0],[0,0.01,0],[0,0,0]])
        self.r = jnp.array([[0.]])

    def initialize_parameters(self, params, ts):
        self.k0 = 7.2 * 10e10 #Arrehnius pre-exponential
        self.E_a = 72_750 #activation energy
        self.R = 8.314 #gas constant
        self.k = lambda T: self.k0 * jnp.exp(-self.E_a / (self.R * T))

        self.Vol = 100 #reactor volume

        rho = 1000 # density
        C = 0.239 #heat capacity
        self.Cp = rho * C

        self.enthalpy = 50_000 #enthalpy of reaction
        self.UA = 50_000 #heat transfer coefficient

        self.q = 100 #fleed flowrate
        self.cf = 1.0 #feed concentration
        self.Tf = 350 #feed temperature

        self.Tcf = 200 #coolant feed temperature
        self.Vol_c = 20 #cooling jacket volume

        self.G = jnp.eye(self.n_var)*jnp.array([50, 50, 0.1])
        self.V = self.sigma*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)*(jnp.array([10,10,0.1])[:self.n_obs])

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval= self.init_lower_bounds, maxval= self.init_upper_bounds)
        targets = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=250, maxval=350)
        return x0, targets

    def sample_params(self, batch_size, mode, ts, key):
        return jnp.zeros(batch_size)
    
    def drift(self, t, state, args):
        Tc, T, c = state
        control = jnp.squeeze(args)

        dc = (self.q / self.Vol) * (self.cf - c) - self.k(T) * c
        dT = (self.q / self.Vol) * (self.Tf - T) - self.enthalpy / (self.Cp) * self.k(T) * c \
                + self.UA / (self.Cp * self.Vol) * (Tc - T)
        dTc = (control / self.Vol_c) * (self.Tcf - Tc) + self.UA / (self.Cp * self.Vol_c) * (T - Tc)
        return jnp.array([dTc, dT, dc])

    def diffusion(self, t, state, args):
        return self.V

    def fitness_function(self, state, control, target, ts):
        u_d = jnp.array([0])
        x_d = jnp.array([0,jnp.squeeze(target),0])
        costs = jax.vmap(lambda _state, _u: (_state-x_d).T @ self.Q @ (_state-x_d) + (_u-u_d)@self.r@(_u-u_d))(state, control)
        return jnp.sum(costs)

    def terminate_event(self, state, **kwargs):
        return jnp.any(jnp.isnan(state.y)) | (state.y[2] > 1) | (state.y[2] < 0) 