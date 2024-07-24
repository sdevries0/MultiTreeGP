import jax.numpy as jnp
import jax
import jax.random as jrandom

from MultiTreeGP.environments.control_environments.control_environment_base import EnvironmentBase

class StirredTankReactor(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs = 3, n_targets = 1):
        self.process_noise = process_noise
        self.obs_noise = obs_noise
        self.n_var = 3
        self.n_control = 1
        self.n_dim = 1
        self.n_targets = n_targets
        self.init_lower_bounds = jnp.array([275, 350, 0.5])
        self.init_upper_bounds = jnp.array([300, 375, 1.0])
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control, self.n_dim, n_obs)

        self.Q = jnp.array([[0,0,0],[0,0.01,0],[0,0,0]])
        self.r = jnp.array([[0.0001]])

    def initialize_parameters(self, params, ts):
        Vol, Cp, dHr, UA, q, Tf, Tcf, Volc = params
        self.Ea  = 72750     # activation energy J/gmol
        self.R   = 8.314     # gas constant J/gmol/K
        self.k0  = 7.2e10    # Arrhenius rate constant 1/min
        self.Vol = Vol       # Volume [L]
        self.Cp  = Cp        # Heat capacity [J/g/K]
        self.dHr = dHr       # Enthalpy of reaction [J/mol]
        self.UA  = UA        # Heat transfer [J/min/K]
        self.q = q           # Flowrate [L/min]
        self.Cf = 1.0        # Inlet feed concentration [mol/L]
        self.Tf  = Tf        # Inlet feed temperature [K]
        self.Tcf = Tcf       # Coolant feed temperature [K]
        self.Volc = Volc       # Cooling jacket volume

        self.k = lambda T: self.k0*jnp.exp(-self.Ea/self.R/T)

        self.G = jnp.eye(self.n_var)*jnp.array([6, 6, 0.05])
        self.V = self.process_noise*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)*(jnp.array([15,15,0.1])[:self.n_obs])

    def sample_params(self, batch_size, mode, ts, key):
        if mode=="Constant":
            Vol = 100*jnp.ones(batch_size)
            Cp = 239*jnp.ones(batch_size)
            dHr = -5.0e4*jnp.ones(batch_size)
            UA = 5.0e4*jnp.ones(batch_size)
            q = 100*jnp.ones(batch_size)
            Tf = 300*jnp.ones(batch_size)
            Tcf = 300*jnp.ones(batch_size)
            Volc = 20.0*jnp.ones(batch_size)
        elif mode=="Different":
            keys = jrandom.split(key, 8)
            Vol = jrandom.uniform(keys[0],(batch_size,),minval=75,maxval=150)
            Cp = jrandom.uniform(keys[1],(batch_size,),minval=200,maxval=350)
            dHr = jrandom.uniform(keys[2],(batch_size,),minval=-55000,maxval=-45000)
            UA = jrandom.uniform(keys[3],(batch_size,),minval=25000,maxval=75000)
            q = jrandom.uniform(keys[4],(batch_size,),minval=75,maxval=125)
            Tf = jrandom.uniform(keys[5],(batch_size,),minval=300,maxval=350)
            Tcf = jrandom.uniform(keys[6],(batch_size,),minval=250,maxval=300)
            Volc = jrandom.uniform(keys[7],(batch_size,),minval=10,maxval=30)
        return (Vol, Cp, dHr, UA, q, Tf, Tcf, Volc)

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval= self.init_lower_bounds, maxval= self.init_upper_bounds)
        targets = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=400, maxval=500)
        return x0, targets
    
    def f_obs(self, key, t_x):
        _, out = super().f_obs(key, t_x)
        # out = jnp.array([out[0], out[1], jnp.clip(out[2], 0, 1)])[:self.n_obs]
        return key, out
    
    def drift(self, t, state, args):
        Tc, T, c = state
        control = jnp.squeeze(args)
        control = jnp.clip(control, 0, 300)
        state = jnp.array([state[0], state[1], jnp.clip(state[2], 0, 1)])

        dc = (self.q/self.Vol)*(self.Cf - c) - self.k(T)*c
        dT = (self.q/self.Vol)*(self.Tf - T) + (-self.dHr/self.Cp)*self.k(T)*c + (self.UA/self.Vol/self.Cp)*(Tc - T)
        dTc = (control/self.Volc)*(self.Tcf - Tc) + (self.UA/self.Volc/self.Cp)*(T - Tc)
        return jnp.array([dTc, dT, dc])

    def diffusion(self, t, state, args):
        return self.V

    def fitness_function(self, state, control, targets, ts):

        x_d = jnp.array([0,jnp.squeeze(targets),0])
        costs = jax.vmap(lambda _state, _u: (_state-x_d).T @ self.Q @ (_state-x_d) + (_u)@self.r@(_u))(state, control)
        return jnp.sum(costs)

    def terminate_event(self, state, **kwargs):
        return jnp.any(jnp.isnan(state.y))# | (state.y[2] > 1) | (state.y[2] < 0) 