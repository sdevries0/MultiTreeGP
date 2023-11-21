import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import diffrax
import genetic_programming as gp
import sympy
import copy
import numpy as np

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
    return lax.bitcast_convert_type(val, new_type)

class SHO:
    def __init__(self, key, dt, sigma, obs_noise, n_obs):
        self.n_obs = n_obs
        self.n_var = 2
        self.n_control = 1
        self.sigma = sigma
        self.obs_noise = obs_noise

        self.q = self.r = 0.5
        self.Q = jnp.array([[self.q,0],[0,0]])
        self.R = jnp.array([[self.r]])
        
        self.key = key
        self.dt = dt

    def initialize(self, params):
        omega, zeta = params
        self.A = jnp.array([[0,1],[-(omega**2),-zeta]])
        self.b = jnp.array([[0.0,1.0]]).T
        self.G = jnp.array([[0,0],[0,1]])
        self.V = self.sigma*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.ones(self.n_obs)

    def f_obs(self,t,x):
        key = jrandom.fold_in(self.key, force_bitcast_convert_type(t))
        return self.C@x + jrandom.normal(key, shape=(self.n_obs,))*self.W

    def compute_riccati(self):
        def riccati(t,S,args):
            return self.A.T@S + S@self.A - S@self.b@(1/self.R)@self.b.T@S + self.Q

        sol = diffrax.diffeqsolve(
                    diffrax.ODETerm(riccati), diffrax.Euler(), 0,10, 0.01, self.Q, max_steps=16**4
                )
        L = 1/self.R@self.b.T@sol.ys[0]
        return L[0]

    def drift(self, t, state, args):
        return self.A@state + self.b@args
    
    def diffusion(self, t, state, args):
        return self.V
    
    def fitness_function(self, state, u, target):
        x_d = jnp.array([target,0])
        u_d = -jnp.linalg.pinv(self.b)@self.A@x_d
        costs = jax.vmap(lambda _state, _u: (_state-x_d).T@self.Q@(_state-x_d) + (_u-u_d)@self.R@(_u-u_d))(state,u)
        return jnp.cumsum(costs)*self.dt
    
    def update_thresholds(self):
        pass

class CartPole:
    def __init__(self, key, dt, sigma, obs_noise, n_obs):
        self.n_obs = n_obs
        self.n_var = 4
        self.n_control = 1
        self.sigma = sigma
        self.obs_noise = obs_noise

        self.threshold_theta = 90 * 2 * jnp.pi / 360
        self.threshold_x = 20
        self.decay = 0.97
        
        self.key = key
        self.dt = dt

    def update_thresholds(self):
        self.threshold_theta *= self.decay
        self.threshold_x *= self.decay

    def initialize(self, params):
        _ = params
        self.g = 9.81
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.cart_mass = 1
        
        self.G = jnp.array([[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]])
        self.V = self.sigma*self.G

        self.C = jnp.eye(self.n_var)[:,:self.n_obs]
        self.W = self.obs_noise*jnp.ones(self.n_obs)

    def f_obs(self,t,state):
        key = jrandom.fold_in(self.key, force_bitcast_convert_type(t))
        return self.C@state + jrandom.normal(key, shape=(self.n_obs,))*self.W

    def drift(self, t, state, args):
        control = jnp.squeeze(args)
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
        # costs = jax.vmap(lambda _state, _u: (_state[0:2]-x_d).T@self.Q@(_state[0:2]-x_d) + (_state[2:4])@self.S@(_state[2:4]) + (_u)@self.R@(_u))(state,u)*self.dt
        # angle_rewards = jnp.cos(state[:,2])>self.threshold_theta
        angle_rewards = jnp.abs(state[:,2])<self.threshold_theta
        position_rewards = jnp.abs(state[:,0])<self.threshold_x

        costs = jnp.cumsum(1.0-angle_rewards*position_rewards)# - 1e5*jnp.square(u))
        return costs
    
class LQG:
    def __init__(self, env, mu0, P0):
        self.env = env
        self.mu = mu0

        self.P = P0*self.env.obs_noise

    def solve(self, x0, ts, key, target, params):
        env = copy.copy(self.env)
        env.initialize(params)

        def drift(t, variables, args):
            x_star, u_star = args
            x = variables[:2]
            mu = variables[2:4]
            P = variables[4:].reshape(2,2)
            
            y = env.f_obs(t,x)
            u = jnp.array([-self.L@(mu-x_star) + u_star[0]])
            K = P@env.C.T*(1/env.W)
            # jax.debug.print("L={L}, {Lx}, {Lx2}",L=self.L,Lx= self.L@(mu-x_star), Lx2=self.L@(mu-x_star)*10)
            dx = env.drift(t,x,u)
            dmu = env.A@mu + env.b@u + K@(y-env.C@mu)
            dP = env.A@P + P@env.A.T-K@env.C@P+env.G*env.sigma@env.G
            return jnp.concatenate([dx, dmu, jnp.ravel(dP)])

        #apply process noise only on x
        def diffusion(t, variables, args):
            x = variables[:2]
            return jnp.concatenate([env.diffusion(t,x,args),jnp.zeros((2,2)),jnp.zeros((4,2))])
        
        solver = diffrax.EulerHeun()
        dt0 = 0.005
        saveat = diffrax.SaveAt(ts=ts)

        self.L = env.compute_riccati()

        x_star = jnp.array([target,-0])
        u_star = -jnp.linalg.pinv(env.b)@env.A@x_star 

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.env.n_var,), key=key) #define process noise
        system = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))

        init = jnp.concatenate([x0, self.mu, jnp.ravel(self.P)])

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, init, saveat=saveat, args=(x_star, u_star), adjoint=diffrax.DirectAdjoint(), max_steps=16**7
        )

        x = sol.ys[:,:2]
        mu = sol.ys[:,2:4]
        u = jax.vmap(lambda m: -self.L@(m-x_star) + u_star[0])(mu)
        y = jax.vmap(env.f_obs)(ts, x)

        costs = env.fitness_function(x, u, target)

        return x, y, u, mu, costs
    
class LQR:
    def __init__(self, env):
        self.env = env

    def solve(self, x0, ts, key, target, params):
        env = copy.copy(self.env)
        def drift(t, x, args):
            y = env.f_obs(t,x)
            u = args[1]-self.L@(y-args[0])
            dx = env.drift(t,x,u)
            return dx
            
        #apply process noise only on x
        def diffusion(t, variables, args):
            x = variables[:2]
            return env.diffusion(t,x,args)
    
        solver = diffrax.Euler()
        dt0 = 0.005
        saveat = diffrax.SaveAt(ts=ts)

        env.initialize(params)
        self.L = env.compute_riccati()

        x_star = jnp.array([target,0])
        u_star = -jnp.linalg.pinv(env.b)@env.A@x_star 

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(2,), key=key) #define process noise
        system = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))

        init = x0

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, init, saveat=saveat, args=(x_star, u_star), adjoint=diffrax.DirectAdjoint(), max_steps=16**5
        )

        x = sol.ys
        
        y = jax.vmap(env.f_obs)(ts, x)
        u = u_star-self.L@(y-x_star).T

        costs = env.fitness_function(x, u, target)

        return x, y, u, costs
    
def evaluate_lqr(data, env):
    x0, ts, targets, noise_keys, params = data
    
    LQR_control = LQR(env)

    xs_lqr, ys_lqr, us_lqr, costs = jax.vmap(LQR_control.solve, in_axes=[0, None, 0, 0, 0])(x0, ts, noise_keys, targets, params)

    # fig, ax = plt.subplots(ncols=x0.shape[0],nrows=1, figsize=(15,4))
    # ax = ax.ravel()
    # for index in range(x0.shape[0]):
    #     ax[index].plot(ts,xs_lqr[index,:,0], label='$x_1$', color='blue')
    #     ax[index].plot(ts,xs_lqr[index,:,1], label='$x_2$', color='red')
    #     ax[index].plot(ts,ys_lqr[index,:,0], label='$y_1$', alpha=0.5, color='blue')
    #     ax[index].plot(ts,ys_lqr[index,:,1], label='$y_2$', alpha=0.5, color='red')
    #     ax[index].plot(ts,us_lqr[index], label='u', color='green')

    # ax[-1].legend()
    # plt.show()

    return costs, jnp.concatenate([xs_lqr, ys_lqr, jnp.atleast_3d(us_lqr)],axis=2)

def evaluate_lqg(data, env, sample_params):
    mu0, P0 = sample_params
    x0, ts, targets, noise_keys, params = data

    LQG_control = LQG(env, mu0, P0)
    
    xs_lqg, ys_lqg, us_lqg, mu_lqg, costs = jax.vmap(LQG_control.solve, in_axes=[0, None, 0, 0, 0])(x0, ts, noise_keys, targets, params)

    # fig, ax = plt.subplots(ncols=x0.shape[0],nrows=1, figsize=(15,4))
    # ax = ax.ravel()
    # for index in range(x0.shape[0]):
    #     ax[index].plot(ts,xs_lqg[index,:,0], label='$x_1$', color='blue')
    #     ax[index].plot(ts,xs_lqg[index,:,1], label='$x_2$', color='red')
    #     ax[index].plot(ts,ys_lqg[index,:,0], label='$y_1$', alpha=0.5, color='blue')
    #     if env.n_obs==2:
    #         ax[index].plot(ts,ys_lqg[index,:,1], label='$y_2$', alpha=0.5, color='red')
    #     ax[index].plot(ts,us_lqg[index], label='u', color='green')
    # ax[-1].legend()
    # plt.show()

    return costs, (xs_lqg, ys_lqg, us_lqg)

def get_data(N, key, n_var, dt, T, mu0, P0):
    init_key, target_key, noise_key, param1_key, param2_key = jrandom.split(key, 5)
    x0 = mu0 + jrandom.normal(init_key, shape=(N,n_var))@P0
    targets = jrandom.uniform(target_key, shape=(N,), minval=-10, maxval=10)
    noise_keys = jrandom.split(noise_key, N)
    omegas = jrandom.uniform(param1_key, shape=(N,), minval=0.5, maxval=1.5)
    zetas = jrandom.uniform(param2_key, shape=(N,), minval=0.0, maxval=0.5)
    ts = jnp.arange(0,T,dt)
    # omegas = jnp.ones(N)
    params = omegas, zetas
    return x0, ts, targets, noise_keys, params

T = 150
dt = 0.5

sigma = 0.01
obs_noise = 0.1

# mu0 = jnp.array([0,0,0,0])
# P0 = jnp.eye(4)*jnp.array([0.05,0.01,0.05,0.01])

mu0 = jnp.zeros(2)
P0 = jnp.eye(2)

generations = 50
n_seeds = 1

best_fitnesses = []
best_solutions = []
rs_best_fitnesses = []
rs_best_solutions = []
all_costs_lqr = []
all_costs_lqg = []

for seed in range(0,n_seeds):
    key = jrandom.PRNGKey(seed)
    env_key, data_key, gp_key = jrandom.split(key, 3)

    env = SHO(env_key, dt, sigma, obs_noise, n_obs=1)
    # env = CartPole(env_key, dt, sigma, obs_noise, n_obs=4)

    data = get_data(50, data_key, env.n_var, dt, T, mu0, P0)

    costs_lqg, state_obs_control_lqg = evaluate_lqg(data, env, (mu0, P0))
    final_cost_lqg=jnp.mean(costs_lqg[:,-1], axis=0)*dt
    all_costs_lqg.append(final_cost_lqg)
    
    print("Optimal cost LQG: ", final_cost_lqg)
    gp = gp.ODE_GP(seed, state_size=1, population_size=200, num_populations=4, max_depth=8, max_init_depth=4)

    best_fitness, best_solution, final_population = gp.run(env, data, generations, pool_size=10, converge_value=0)
#     # print("random search")
#     # rs_best_fitness, rs_best_solution, rs_final_population = gp.random_search(env, data, generations, pool_size=10, converge_value=converge_value)
    
    best_fitnesses.append(best_fitness)
    best_solutions.append(best_solution)
    # rs_best_fitnesses.append(rs_best_fitness)
    # rs_best_solutions.append(rs_best_solution)
    
best_fitnesses = jnp.array(best_fitnesses).reshape(n_seeds, generations)

# rs_best_fitnesses = jnp.array(rs_best_fitnesses).reshape(n_seeds, generations)

name = 'obs_random_env'
np.save(f'data_files/{name}/best_fitnesses.npy',best_fitnesses)
np.save(f'data_files/{name}/rs_best_fitnesses.npy',rs_best_fitnesses)
np.save(f'data_files/{name}/best_solutions.npy',best_solutions)
np.save(f'data_files/{name}/lqg.npy',jnp.array(all_costs_lqg))