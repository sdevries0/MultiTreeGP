import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
import diffrax
import copy
from typing import Tuple, Sequence
from pathos.multiprocessing import ProcessingPool as Pool
import time

from expression import Expression
from environments.cart_pole import CartPole
from environments.stochastic_harmonic_oscillator import StochasticHarmonicOscillator
from networkTrees import NetworkTrees
import reproduction, miscellaneous, simplification, migration

def define_expressions(state_size, obs_size, control_size):
    binary_operators_map = {"+":lambda a,b:a+b, "-":lambda a,b:a-b, "*":lambda a,b:a*b, "/":lambda a,b:a/b, "**":lambda a,b:a**b}
    unary_operators_map = {"sin":lambda a:jnp.sin(a), "cos":lambda a:jnp.cos(a)}#,"exp":lambda a:jnp.clip(jnp.exp(a),a_max=100), "sqrt":lambda a:jnp.sqrt(jnp.abs(a))}

    variables = ["y" + str(i) for i in range(obs_size)]
    state_variables = ["a" + str(i) for i in range(state_size)]
    control_variables = ["u" + str(i) for i in range(control_size)]

    return Expression(binary_operators_map, unary_operators_map, variables, state_variables, control_variables)

class Evaluator:
    def __init__(self, env, expressions, state_size, key, dt, T, parsimony = 0.2, batch_size = 32):
        self.env = env
        self.expressions = expressions
        self.state_size = state_size
        self.latent_size = self.env.n_var
        self.parsimony = parsimony

        self.data = self.get_data(batch_size, key, self.latent_size, dt, T, env.mu0, env.P0)

    def get_data(self, batch_size, key, n_var, dt, T, mu0, P0):
        init_key, target_key, noise_key, param1_key, param2_key = jrandom.split(key, 5)
        x0 = mu0 + jrandom.normal(init_key, shape=(batch_size,n_var))@P0
        targets = jrandom.uniform(target_key, shape=(batch_size,), minval=-10, maxval=10)
        noise_keys = jrandom.split(noise_key, batch_size)
        omegas = jrandom.uniform(param1_key, shape=(batch_size,), minval=0.5, maxval=1.5)
        zetas = jrandom.uniform(param2_key, shape=(batch_size,), minval=0.0, maxval=0.5)
        ts = jnp.arange(0,T,dt)
        # omegas = jnp.ones(N)
        params = omegas, zetas
        return x0, ts, targets, noise_keys, params

    #Evaluation methods
    def evaluate_tree(self, tree: list):
        "A tree is transformed to a callable function, represented as nested lamba functions"
        if tree[0] in self.expressions.binary_operators:
            assert len(tree) == 3, f"The operator {tree[0]} requires two inputs"
            left = self.evaluate_tree(tree[1])
            right = self.evaluate_tree(tree[2])
            return lambda x, a, u, t: self.expressions.binary_operators_map[tree[0]](left(x,a,u,t),right(x,a,u,t))
        
        elif tree[0] in self.expressions.unary_operators:
            assert len(tree) == 2, f"The operator {tree[0]} requires one input"
            left = self.evaluate_tree(tree[1])
            return lambda x, a, u, t: self.expressions.unary_operators_map[tree[0]](left(x,a,u,t))

        assert len(tree) == 1, "Leaves should not have children"
        if isinstance(tree[0],jax.numpy.ndarray):
            return lambda x, a, u, t: tree[0]
        elif tree[0]=="target":
            return lambda x, a, u, t: t
        elif tree[0] in self.expressions.state_variables:
            return lambda x, a, u, t: a[self.expressions.state_variables.index(tree[0])]
        elif tree[0] in self.expressions.variables:
            return lambda x, a, u, t: x[self.expressions.variables.index(tree[0])]
        elif tree[0] in self.expressions.control_variables:
            return lambda x, a, u, t: u[self.expressions.control_variables.index(tree[0])]
        print(tree)
    
    def evaluate_trees(self, trees: NetworkTrees):
        "Evaluate the trees in the network"
        return [self.evaluate_tree(tree) for tree in trees()]
   
    def evaluate_control_loop(self, model: Tuple, x0: Sequence[float], ts: Sequence[float], target: float, key: PRNGKey, params: Tuple):
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
        env.initialize_parameters(params)

        state_equation, readout = model

        #Define state equation
        def _drift(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]

            # jax.debug.print("P={P}", P=a[2:])

            y = env.f_obs(t, x) #Get observations from system
            u = readout(y, a, 0, target) #Readout control from hidden state
            # u = a
            
            dx = env.drift(t, x, u) #Apply control to system and get system change
            da = state_equation(y, a, u, target) #Compute hidden state updates
            return jnp.concatenate([dx, da])
        
        #Define diffusion
        def _diffusion(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]
            y = env.f_obs(t, x)
            u = jnp.array([readout(y, a, 0, target)])
            # u = a
            return jnp.concatenate([env.diffusion(t, x, u), jnp.zeros((self.state_size, self.latent_size))]) #Only the system is stochastic
        
        solver = diffrax.Euler()
        dt0 = 0.005
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])
        # _x0 = jnp.concatenate([x0, jnp.zeros(2), jnp.array([1,0,0,1])*self.env.obs_noise])

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=key)
        system = diffrax.MultiTerm(diffrax.ODETerm(_drift), diffrax.ControlTerm(_diffusion, brownian_motion))

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=16**4
        )
        xs = sol.ys[:,:self.latent_size]
        ys = jax.vmap(env.f_obs)(ts, xs) #Map states to observations
        activities = sol.ys[:,self.latent_size:]
        
        us = jax.vmap(readout, in_axes=[0,0,None,None])(ys, activities, jnp.array([0]), target) #Map hidden state to control 
        # us = activities   
        # fitness = jax.lax.cond(jnp.isnan(us).any(),lambda x, u, t: 1e3*jnp.ones(ts.shape), lambda x, u, t: env.fitness_function(x, u, t), xs, us, target) #Compute fitness with cost function in the environment
        fitness = env.fitness_function(xs, us, target)

        return xs, ys, us, activities, fitness

    def get_fitness(self, model: NetworkTrees, add_regularization: bool = True):
        "Determine the fitness of a tree"
        _, _, _, _, fitness = self.evaluate_model(model)

        fitness = jnp.mean(fitness[:,-1])
        
        if jnp.isinf(fitness) or jnp.isnan(fitness):
            fitness = jnp.array(1e6)
        if add_regularization: #Add regularization to punish trees for their size
            return jnp.clip(fitness + self.parsimony*len(jax.tree_util.tree_leaves(model)),0,1e6)
        else:
            return jnp.clip(fitness,0,1e6)
        
    def evaluate_model(self, model: NetworkTrees):
        "Evaluate a tree by simulating the environment and controller as a coupled system"
        
        #Trees to callable functions
        tree_funcs = self.evaluate_trees(model)
        state_equation = jax.jit(lambda y, a, u, tar: jnp.array([tree_funcs[i](y, a, u, tar) for i in range(self.state_size)]))
        readout_layer = self.evaluate_tree(model.readout_tree)
        readout = lambda y, a, _, tar: jnp.atleast_1d([readout_layer(y, a, _, tar)])
        model = (state_equation, readout)

        x0, ts, targets, noise_keys, params = self.data
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0])(model, x0, ts, targets, noise_keys, params) #Run coupled differential equations of state and control and get fitness of the model
    
key = jrandom.PRNGKey(0)
env_key, data_key = jrandom.split(key)

population_size = 100
num_populations = 4
num_generations = 5
state_size = 3
T = 150
dt = 0.5
pool_size = 10
continue_population = None
restart_iter_threshold = jnp.array([10,8,6,4])
migration_period = 5

sigma = 0.01
obs_noise = 0.1

env = StochasticHarmonicOscillator(env_key, sigma, obs_noise, n_obs=2)
expressions = define_expressions(state_size, env.n_obs, env.n_control)
evaluator = Evaluator(env, expressions, state_size, data_key, dt, T)
reproducer = reproduction.Reproducer(expressions, population_size, state_size, num_populations)

n_seeds = 1

best_fitnesses = jnp.zeros((n_seeds, num_generations))
best_solutions = []
# rs_best_fitnesses = []
# rs_best_solutions = []
# all_costs_lqr = []
# all_costs_lqg = []

for seed in range(0,n_seeds):
    pool = Pool(pool_size)  
    pool.close()

    best_solution = None
    best_fitness = jnp.inf
    best_fitness_per_population = jnp.zeros((num_generations, num_populations))

    last_restart = jnp.zeros(num_populations)

    if continue_population is None:
        #Initialize new population
        key, new_key = jrandom.split(key)
        populations = reproducer.sample_trees(new_key, population_size, num_populations=num_populations)
    else:
        #Continue from a previous population
        populations = continue_population
    
    for g in range(num_generations):
        pool.restart()
        fitness = pool.amap(lambda x: evaluator.get_fitness(x),miscellaneous.flatten(populations)) #Evaluate each solution parallely on a pool of workers
        pool.close()
        pool.join()
        tries = 0
        while not fitness.ready():
            time.sleep(1)
            tries += 1

            if tries >= 200:
                print("TIMEOUT")
                break

        flat_fitnesses = jnp.array(fitness.get())
        fitnesses = jnp.reshape(flat_fitnesses,(num_populations,population_size))
        # print(jnp.mean(fitnesses, axis=1),jnp.min(fitnesses, axis=1))

        best_fitness_per_population = best_fitness_per_population.at[g].set(jnp.min(fitnesses, axis=1))
        
        #Set the fitness of each solution
        for pop in range(num_populations):
            population = populations[pop]
            for candidate in range(population_size):
                population[candidate].set_fitness(fitnesses[pop,candidate])

        best_solution_of_g = miscellaneous.best_solution(populations, fitnesses)
        best_fitness_at_g = evaluator.get_fitness(best_solution_of_g, add_regularization=False)
        if best_fitness_at_g < best_fitness:
            best_fitness = best_fitness_at_g
            best_solution = best_solution_of_g
        best_fitnesses = best_fitnesses.at[seed, g].set(best_fitness)

        if best_fitnesses[seed, g]==best_fitnesses[seed, g-2]: #A solution reached a satisfactory score
            best_solution_string = simplification.trees_to_sympy(best_solution)
            print(f"Converge settings satisfied, best fitness {best_fitness}, best solution: {best_solution_string}, readout: {simplification.tree_to_sympy(best_solution.readout_tree)}")
            best_fitnesses = best_fitnesses.at[g:].set(best_fitness)

            break

        elif g < num_generations-1: #The final generation has not been reached yet, so a new population is sampled
            best_solution_string = simplification.trees_to_sympy(best_solution)
            print(f"In generation {g+1}, average fitness: {jnp.mean(fitnesses)}, best_fitness: {best_fitness}, best solution: {best_solution_string}, readout: {simplification.tree_to_sympy(best_solution.readout_tree)}")

            #Migrate individuals between populations every few generations
            if ((g+1)%migration_period)==0:
                populations = migration.migrate_populations(populations)

            last_restart = last_restart + 1
                            
            for pop in range(num_populations):
                #Generate new population
                key, new_key = jrandom.split(key)
                if last_restart[pop]>restart_iter_threshold[pop] and best_fitness_per_population[g,pop] >= best_fitness_per_population[g-restart_iter_threshold[pop],pop]:
                    print(f"Stuck restart in population {pop+1}")
                    populations[pop] = reproducer.sample_trees(new_key, population_size, num_populations=1)[0]
                    last_restart = last_restart.at[pop].set(0)
                else:
                    populations[pop] = reproducer.next_population(populations[pop], jnp.mean(flat_fitnesses), pop, new_key)
            
        else: #Final generation is reached
            best_solution_string = simplification.trees_to_sympy(best_solution)
            print(f"Final generation, average fitness: {jnp.mean(fitnesses)}, best_fitness: {best_fitness}, best solution: {best_solution_string}, readout: {simplification.tree_to_sympy(best_solution.readout_tree)}")