import jax
import jax.numpy as jnp
import jax.random as jrandom
import time
from jax.random import PRNGKey
from typing import Sequence, Tuple
import copy
import diffrax
from miscellaneous.networks import NetworkTrees
import genetic_operators.simplification as simplification

class Evaluator:
    def __init__(self, env, expressions, layer_sizes, key, dt, T, parsimony = 0.1, batch_size = 32, param_setting: str = "Constant"):
        self.env = env
        self.expressions = expressions
        self.state_size = layer_sizes[0]
        self.control_size = layer_sizes[1]
        self.latent_size = self.env.n_var*self.env.n_dim
        self.parsimony = parsimony

        self.ts, self.x0s, self.targets, self.process_noise_keys, self.obs_noise_keys, self.params = self.get_data(batch_size, env, key, dt, T, param_setting)

        self.constant_fitness = 10#self.get_fitness(NetworkTrees([[[jnp.array(1)] for _ in range(self.state_size)], [[jnp.array(1)] for _ in range(self.control_size)]]), add_regularization=False)

    def get_data(self, batch_size, env, key, dt, T, param_setting):
        init_key, target_key, noise_key1, noise_key2, param_key = jrandom.split(key, 5)
        x0, targets = env.sample_init_states(batch_size, init_key)
        process_noise_keys = jrandom.split(noise_key1, batch_size)
        obs_noise_keys = jrandom.split(noise_key2, batch_size)
        ts = jnp.arange(0,T,dt)

        params = env.sample_params(batch_size, param_setting, ts, param_key)
        return ts, x0, targets, process_noise_keys, obs_noise_keys, params

    def compute_ricatti(self, A, b, Q, R):
        def _f(t,x,args):
            return A.T@x + x@A - x@b@jnp.linalg.inv(R)@b.T@x + Q
        
        system = diffrax.ODETerm(_f)
        sol = diffrax.diffeqsolve(
            system, diffrax.Euler(), 0, 10, 0.01, jnp.zeros((2,2)), max_steps=16**4
        )
        return sol.ys[0]

    def evaluate_LQG(self, x0: Sequence[float], ts: Sequence[float], target: float, process_noise_key: PRNGKey, obs_noise_key: PRNGKey, params: Tuple):
        env = copy.copy(self.env)
        env.initialize_parameters(params)

        def drift(t, variables, args):
            x_star, u_star, L2 = args
            x = variables[:self.latent_size]
            mu = variables[self.latent_size:self.latent_size*2]
            P = variables[2*self.latent_size:2*self.latent_size+self.latent_size**2].reshape(self.latent_size,self.latent_size)
            S = variables[2*self.latent_size+self.latent_size**2:].reshape(self.latent_size,self.latent_size)
            
            _, y = env.f_obs(obs_noise_key, (t, x))
            L = jnp.linalg.inv(env.R)@env.b.T@S
            u = jnp.array(-L2@(mu-x_star) + u_star)
            K = P@env.C.T@jnp.linalg.inv(env.W)

            dx = env.drift(t,x,u)
            dmu = env.A@mu + env.b@u + K@(y-env.C@mu)
            dP = env.A@P + P@env.A.T-K@env.C@P+env.V
            dS = env.A.T@S + S@env.A - S@env.b@jnp.linalg.inv(env.R)@env.b.T@S + env.Q

            return jnp.concatenate([dx, dmu, jnp.ravel(dP), jnp.ravel(dS)])

        #apply process noise only on x
        def diffusion(t, variables, args):
            x = variables[:self.latent_size]
            return jnp.concatenate([env.diffusion(t,x,args),jnp.zeros((self.latent_size,self.latent_size)),jnp.zeros((self.latent_size**2,self.latent_size)), jnp.zeros((self.latent_size**2,self.latent_size))])
        
        solver = diffrax.EulerHeun()
        dt0 = 0.005
        saveat = diffrax.SaveAt(ts=ts)

        L2 = jnp.linalg.inv(env.R)@env.b.T@self.compute_ricatti(env.A, env.b, env.Q, env.R)

        #Set target state and control
        x_star = jnp.zeros((self.latent_size))
        for i in range(env.n_dim):
            x_star = x_star.at[i*env.n_var].set(target[i])
        u_star = -jnp.linalg.pinv(env.b)@env.A@x_star

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key) #define process noise
        system = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))

        init = jnp.concatenate([x0, env.mu0, jnp.ravel(env.P0), jnp.zeros((4))])

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, init, saveat=saveat, args=(x_star, u_star, L2), adjoint=diffrax.DirectAdjoint(), max_steps=16**7
        )
        x = sol.ys[:,:self.latent_size]
        mu = sol.ys[:,self.latent_size:2*self.latent_size]
        S = sol.ys[:,self.latent_size*2+self.latent_size**2:].reshape(ts.shape[0],self.latent_size, self.latent_size)
        L = jax.vmap(lambda s: jnp.linalg.inv(env.R)@env.b.T@s)(S)
        u = jax.vmap(lambda m, l: -l@(m-x_star) + u_star, in_axes=[0,None])(mu, L2) #Map states to control
        _, y = jax.lax.scan(env.f_obs, obs_noise_key, (ts, x)) #Map states to observations

        costs = env.fitness_function(x, u, target)

        return x, y, u, costs
    
    def LQG_fitness(self):
        x, y, u, costs = jax.vmap(self.evaluate_LQG, in_axes=[0, None, 0, 0, 0, 0])(self.x0s, self.ts, self.targets, self.process_noise_keys, self.obs_noise_keys, self.params)
        
        return x, y, u, jnp.mean(costs[:,-1])
   
    def evaluate_control_loop(self, model: Tuple, x0: Sequence[float], ts: Sequence[float], target: float, process_noise_key: PRNGKey, obs_noise_key: PRNGKey, params: Tuple):
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation of the control is defined by the set of trees
        Inputs:
            model (NetworkTrees): Model with trees for the hidden state and readout
            x0 (float): Initial state of the system
            ts (Array[float]): time points on which the controller is evaluated
            target (float): Target position that the system should reach
            key (PRNGKey)
            params (Tuple[float]): Parameters that define the system

        Returns:
            xs (Array[float]): States of the system at every time point
            ys (Array[float]): Observations of the system at every time point
            us (Array[float]): Control of the model at every time point
            activities (Array[float]): Activities of the hidden state of the model at every time point
            fitness (float): Fitness of the model 
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        state_equation, readout = model

        #Define state equation
        def _drift(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]

            _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
            # jax.debug.print("P={P}", P=y)
            u = readout({"y":y, "a":a, "tar":target}) #Readout control from hidden state

            # u = a
            dx = env.drift(t, x, u) #Apply control to system and get system change
            da = state_equation({"y":y, "a":a, "u":u, "tar":target}) #Compute hidden state updates
            return jnp.concatenate([dx, da])
        
        #Define diffusion
        def _diffusion(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]
            _, y = env.f_obs(obs_noise_key, (t, x))
            u = readout({"y":y, "a":a, "tar":target})
            # u = a
            return jnp.concatenate([env.diffusion(t, x, u), jnp.zeros((self.state_size, self.latent_size))]) #Only the system is stochastic
        
        solver = diffrax.EulerHeun()
        dt0 = 0.005
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])
        # _x0 = jnp.concatenate([x0, jnp.zeros(2), jnp.array([1,0,0,1])*self.env.obs_noise])

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key)
        system = diffrax.MultiTerm(diffrax.ODETerm(_drift), diffrax.ControlTerm(_diffusion, brownian_motion))
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=16**4, discrete_terminating_event=self.env.terminate_event
        )

        xs = sol.ys[:,:self.latent_size]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        activities = sol.ys[:,self.latent_size:]
        us = jax.vmap(lambda y, a, t: readout({"y":y, "a":a, "tar":target}), in_axes=[0,0,None])(ys, activities, target)

        fitness = env.fitness_function(xs, us, target, ts)

        return xs, ys, us, activities, fitness

    def get_fitness(self, model: NetworkTrees, add_regularization: bool = True):
        "Determine the fitness of a tree"
        if model.fitness != None and add_regularization:
            return model.fitness

        # if sum([leaf in self.expressions.state_variables for leaf in jax.tree_util.tree_leaves(model.readout_trees)])==0 and add_regularization:
        #     return self.constant_fitness
        
        _, _, _, _, fitness = self.evaluate_model(model)

        fitness = jnp.mean(fitness)
        
        if jnp.isinf(fitness) or jnp.isnan(fitness):
            fitness = jnp.array(1e6)
        if add_regularization: #Add regularization to punish trees for their size
            return jnp.clip(fitness + self.parsimony*len(jax.tree_util.tree_leaves(model)),0,1e6)
        else:
            return jnp.clip(fitness,0,1e6)
         
    def evaluate_model(self, model: NetworkTrees):
        "Evaluate a tree by simulating the environment and controller as a coupled system"
        
        #Trees to callable functions
        model_functions = model.tree_to_function(self.expressions)

        #Run coupled differential equations of state and control and get fitness of the model
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0])(model_functions, self.x0s, self.ts, self.targets, self.process_noise_keys, self.obs_noise_keys, self.params)