import jax
print(jax.devices())
from jax import Array

import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import optax
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import sympy

from typing import Tuple, Callable
import time

from MultiTreeGP.algorithms.strategy_base import Strategy
from MultiTreeGP.genetic_operators.crossover import crossover_trees
from MultiTreeGP.genetic_operators.initialization import sample_population, sample_tree
from MultiTreeGP.genetic_operators.mutation import initialize_mutation_function
from MultiTreeGP.genetic_operators.reproduction import evolve_populations, evolve_population


def lambda_func_arity1(f):
    return lambda x, y, _data: f(x)

def lambda_func_arity2(f):
    return lambda x, y, _data: f(x, y)

def lambda_leaf(i):
    return lambda x, y, _data: _data[i]

class GeneticProgramming(Strategy):
    """Genetic programming strategy of symbolic expressions.

        Attributes:
            num_generations: The number of generations over which to evolve the population.
            population_size: Number of candidates in the population.
            fitness_function: Function that evaluates a candidate and assigns a fitness.
            current_generation: A reference to keep track of the current generation.
            best_solutions: Best solution at each generation.
            best_fitnesses: Best fitness at each generation.
            expressions: Expressions for each layer in a tree policy.
            layer_sizes: Size of each layer in a tree policy.
            num_populations: Number of subpopulations.
            max_depth: Highest depth of a tree.
            max_init_depth: Highest depth of a tree during initialization.
            init_method: Method for initializing the trees.
            size_parsinomy: Parsimony factor that increases the fitness of a candidate based on its size.
            leaf_sd: Standard deviation to sample constants.
            migration_period: Number of generations after which migration happens.
            migration_method: Method that specifies how migration takes place.
            migration_size: Number of candidates to migrate.
            restart_iter_threshold: Number of generations that a subpopulation did not improve before a restart occurs.
            last_restart: Counter that keeps track of the last restart of each subpopulation.
            tournament_size: Size of the tournament.
            selection_pressures: Selection pressure in a tournament, varying per population.
            tournament_probabilities: Probability of each rank in a tournament to be selected for reproduction.
            reproduction_type_probabilities: Probability of each reproduction method to be selected, varying per population.
            reproduction_probabilities: Probability of a tree to be adapted in a candidate, varying per population.
            elite_percentage: Percentage of elite candidates that procceed to the next population, varying per population.
            mutation_probabilities: Probability for each mutation method to be selected.
            best_fitness_per_population: The best fitness in each subpopulation.
        """
    def __init__(self, num_generations: int, 
                 population_size: int, 
                 fitness_function: Callable, 
                 function_list: list,
                 variable_list: list, 
                 layer_sizes: Array,
                 num_populations: int = 1, 
                 max_depth: int = 7, 
                 max_init_depth: int = 4, 
                 max_nodes: int = 40,
                 tournament_size: int = 7, 
                 size_parsinomy: float = 1.0, 
                 migration_period: int = 5,
                 leaf_sd: float = 1.0, 
                 migration_percentage: float = 0.1,
                 gradient_optimisation: bool = False,
                 gradient_steps: int = 10) -> None:
        
        self.layer_sizes = layer_sizes
        self.num_populations = num_populations
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.max_nodes = min(max_nodes, 2**self.max_depth-1)
        self.num_trees = jnp.sum(self.layer_sizes)
        super().__init__(num_generations, population_size, fitness_function, (self.num_trees, self.max_nodes, 4))

        self.size_parsinomy = size_parsinomy
        self.leaf_sd = leaf_sd

        self.migration_period = migration_period
        self.migration_size = int(migration_percentage*population_size)

        self.gradient_optimisation = gradient_optimisation
        self.gradient_steps = gradient_steps

        self.tournament_size = tournament_size
        self.selection_pressures = jnp.linspace(0.6,0.9,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(0.9,0.4,self.num_populations),jnp.linspace(0.1,0.5,self.num_populations),
                                                           jnp.linspace(0.0,0.1,self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(1.0, 0.5, self.num_populations)
        self.elite_size = int(0.1*self.population_size)
        # self.elite_size = 10

        self.best_fitness_per_population = jnp.zeros((num_generations, self.num_populations))

        self.optimizer = optax.adam(learning_rate=0.001, b1=0.9, b2=0.999)

        self.map_b_to_d = self.create_map_b_to_d(self.max_depth)

        func_dict = {}
        func_to_string = {}
        functions = [lambda x, y, _data: 0.0, lambda x, y, _data: 0.0]

        slots = [0, 0]
        index = 2
        func_probabilities = jnp.zeros(len(function_list))

        for func_tuple in function_list:
            string = func_tuple[0]
            f = func_tuple[1]
            arity = func_tuple[2]
            if len(func_tuple)==4:
                probability = func_tuple[3]
            else:
                probability = 1.0

            if string not in func_dict:
                func_dict[string] = index
                func_to_string[index] = string
                if arity==1:
                    functions.append(lambda_func_arity1(f))
                    slots.append(1)
                elif arity==2:
                    functions.append(lambda_func_arity2(f))
                    slots.append(2)
                func_probabilities = func_probabilities.at[index-2].set(probability)
                index += 1

        self.func_probabilities = func_probabilities
        self.func_indices = jnp.arange(2, index)
        var_start_index = index

        data_index = 0
        assert len(layer_sizes) == len(variable_list), "There is not a set of expressions for every type of layer"

        for var_list in variable_list:
            for var in var_list:
                if var not in func_dict:
                    func_dict[var] = index
                    func_to_string[index] = var
                    functions.append(lambda_leaf(data_index))
                    slots.append(0)
                    index += 1
                    data_index += 1
        
        self.leaf_indices = jnp.arange(var_start_index, index)
        variable_probabilities = jnp.zeros((self.num_trees, data_index))

        counter = 0
        for layer_i, var_list in enumerate(variable_list):
            p = jnp.zeros((data_index))
            for var in var_list:
                p = p.at[func_dict[var] - var_start_index].set(1)

            for _ in range(layer_sizes[layer_i]):
                variable_probabilities = variable_probabilities.at[counter].set(p)
                counter += 1

        self.slots = jnp.array(slots)

        self.func_dict = func_dict
        self.func_to_string = func_to_string
        self.functions = functions
        self.variable_probabilities = variable_probabilities

        # print(self.func_dict, self.func_to_string, len(self.functions))
        # print(self.slots)
        # print(self.func_indices, self.leaf_indices)
        # print(self.func_probabilities, self.variable_probabilities)

        self.sample_args = (self.leaf_indices, self.func_indices, self.func_probabilities, self.slots, self.leaf_sd, self.map_b_to_d)
        self.sample_population = partial(sample_population, 
                                         num_trees = self.num_trees, 
                                         max_init_depth = self.max_init_depth, 
                                         max_depth = self.max_depth, 
                                         max_nodes = self.max_nodes, 
                                         variable_probabilities = self.variable_probabilities,
                                         args = self.sample_args)
        self.sample_tree = partial(sample_tree, max_depth = self.max_depth, max_nodes = self.max_nodes, args = self.sample_args)

        self.mutate_args = (self.sample_tree, self.max_nodes, self.max_init_depth, self.leaf_indices, self.func_indices, self.func_probabilities, self.slots)
        self.mutate_trees = initialize_mutation_function(self.mutate_args)

        self.partial_crossover = partial(crossover_trees, func_indices = self.func_indices, max_nodes = self.max_nodes)

        self.reproduction_functions = [self.partial_crossover, self.mutate_pair, self.sample_pair]

        self.jit_evolve_population = jax.jit(partial(evolve_population, reproduction_functions = self.reproduction_functions, elite_size = self.elite_size, tournament_size = self.tournament_size, num_trees = self.num_trees, population_size=population_size))

        self.partial_ff = partial(self.fitness_function, eval = self.vmap_foriloop)
        self.vmap_trees = jax.vmap(self.partial_ff, in_axes=[0, 0, None])
        self.jit_body_fun = jax.jit(partial(self.body_fun, functions = self.functions))
        self.vmap_grad_ff = jax.vmap(jax.value_and_grad(self.partial_ff), in_axes=[0, 0, None])
        
        devices = mesh_utils.create_device_mesh((len(jax.devices('cpu'))))
        self.mesh = Mesh(devices, axis_names=('i'))

        @partial(shard_map, mesh=self.mesh, in_specs=(P('i'), P(None)), out_specs=(P('i'), P('i')), check_rep=False)
        def shard_eval(array, data):
            result = self.vmap_trees(array[...,3:], array[...,:3], data)
            return result, array
            
        @partial(shard_map, mesh=self.mesh, in_specs=(P('i'), P(None)), out_specs=(P('i'), P('i')), check_rep=False)
        def shard_optimise(array, data):
            result, _array = self.optimise(array, data, self.gradient_steps)
            return result, _array
        
        self.jit_eval = jax.jit(shard_eval)
        self.jit_optimise = jax.jit(shard_optimise)

    def create_map_b_to_d(self, depth):
        max_nodes = 2**depth-1
        current_depth = 0
        map_b_to_d = jnp.zeros(max_nodes)
        for i in range(max_nodes):
            if i>0:
                parent = (i + (i%2) - 2)//2
                value = map_b_to_d[parent]
                if (i % 2)==0:
                    new_value = value + 2**(depth-current_depth+1)
                else:
                    new_value = value + 1
                map_b_to_d = map_b_to_d.at[i].set(new_value)
            current_depth += i==(2**current_depth-1)
        return max_nodes - 1 - map_b_to_d

    def initialize_population(self, key: PRNGKey) -> list:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        keys = jr.split(key, self.num_populations)
        populations = jax.vmap(self.sample_population, in_axes=[0, None])(keys, self.population_size)

        return populations
    
    def tree_to_string(self, tree):
        if tree[-1,0]==1:
            return "{:.2f}".format(tree[-1,3])
        elif tree[-1,1]<0:
            return self.func_to_string[tree[-1,0].astype(int).item()]
        elif tree[-1,2]<0:
            substring = self.tree_to_string(tree[:tree[-1,1].astype(int)+1])
            return f"{self.func_to_string[tree[-1,0].astype(int).item()]}({substring})"
        else:
            substring1 = self.tree_to_string(tree[:tree[-1,1].astype(int)+1])
            substring2 = self.tree_to_string(tree[:tree[-1,2].astype(int)+1])
            return f"({substring1}){self.func_to_string[tree[-1,0].astype(int).item()]}({substring2})"
        
    def to_string(self, trees):
        string_output = ""
        tree_index = 0
        layer_index = 0
        for tree in trees:
            if tree_index==0:
                string_output += "["
            string_output += str(sympy.parsing.sympy_parser.parse_expr(self.tree_to_string(tree)))
            if tree_index < (self.layer_sizes[layer_index] - 1):
                string_output += ", "
                tree_index += 1
            else:
                string_output += "]"
                if layer_index < (self.layer_sizes.shape[0] - 1):
                    string_output += ", "
                tree_index = 0
                layer_index += 1
        return string_output
    
    def body_fun(self, i, carry, functions):
        array, data = carry
        f_idx, a_idx, b_idx, float = array[i]
    
        x = array[a_idx.astype(int), 3]
        y = array[b_idx.astype(int), 3]
        value = jax.lax.select(f_idx == 1, float, jax.lax.switch(f_idx.astype(int), functions, x, y, data))
        
        # array = array.at[i, 3].set(jax.lax.select(jnp.isnan(value) | jnp.isinf(value), 0.0, value))
        array = array.at[i, 3].set(value)

        return (array, data)

    def foriloop(self, data, array):
        # print(array.shape)
        x, _ = jax.lax.fori_loop(0, self.max_nodes, self.jit_body_fun, (array, data))
        return x[-1, -1]

    def vmap_foriloop(self, array, data):
        # _, result = jax.lax.scan(self.foriloop, init=data, xs=array)
        result = jax.vmap(self.foriloop, in_axes=[None, 0])(data, array)
        # print(result.shape)
        return result
       
    def evaluate_population(self, populations: list, data: Tuple) -> Tuple[Array, list]:
        """Evaluates every candidate in population and assigns a fitness.

        :param population: Population of candidates
        :param data: The data required to evaluate the population.

        Returns: Fitness and evaluated population.
        """
        flat_populations = populations.reshape(self.num_populations*self.population_size, *populations.shape[2:])
        flat_populations = jax.device_put(flat_populations, NamedSharding(self.mesh, P('i')))
        
        # start = time.time()
        # fitness, optimised_population = self.jit_optimise(flat_populations, data) if (self.gradient_optimisation & (((self.current_generation+1)%5)==0)) else self.jit_eval(flat_populations, data)
        fitness, optimised_population = self.jit_eval(flat_populations, data)

        if (self.gradient_optimisation & ((self.current_generation+1)%10==0)):
            best_candidates_idx = jnp.argsort(fitness)[:100]
            fitness2, optimised_population2 = self.jit_optimise(flat_populations[best_candidates_idx], data)
            optimised_population = optimised_population.at[best_candidates_idx].set(optimised_population2)
            fitness = fitness.at[best_candidates_idx].set(fitness2)

        fitness = fitness + jax.vmap(lambda array: self.size_parsinomy * jnp.sum(array[:,:,0]!=0))(flat_populations)
        fitness = fitness.reshape((self.num_populations, self.population_size))

        self.best_fitness_per_population = self.best_fitness_per_population.at[self.current_generation].set(jnp.min(fitness, axis=1))
        best_solution_of_g = optimised_population[jnp.argmin(fitness)]
        best_fitness_of_g = jnp.min(fitness)

        #Keep track of best solution
        if self.current_generation == 0:
            best_fitness = best_fitness_of_g
            best_solution = best_solution_of_g
        else:
            best_fitness = self.best_fitnesses[self.current_generation - 1]
            best_solution = self.best_solutions[self.current_generation - 1]

            if best_fitness_of_g < best_fitness:
                best_fitness = best_fitness_of_g
                best_solution = best_solution_of_g
            
        self.best_solutions = self.best_solutions.at[self.current_generation].set(best_solution)
        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)

        # print(fitness[0,0])

        # print("elapsed time:", time.time()-start)

        return fitness, optimised_population.reshape((self.num_populations, self.population_size, *optimised_population.shape[1:]))      
            
    def epoch(self, carry, x):
        trees, states, data = carry
        loss, gradients = self.vmap_grad_ff(trees[...,3:], trees[...,:3], data)

        updates, states = jax.vmap(self.optimizer.update)(gradients, states, trees[...,3])
        new_trees = trees.at[...,3:].set(jax.vmap(lambda t, u: t + u)(trees[...,3:], updates))
        return (new_trees, states, data), (trees, loss)

    def optimise(self, trees, data: Tuple, n_steps):
        """Evaluates a candidate and assigns a fitness and optionally optimises the constants in the tree.

        :param candidate: Candidate solution.
        :param data: The data required to evaluate the population.
        :param optimise: Whether to optimise the constants in the tree.
        :param num_steps: Number of steps during constant optimisation.

        Returns: Optimised and evaluated candidate.
        """
        states = jax.vmap(self.optimizer.init)(trees[...,3:])

        _, out = jax.lax.scan(self.epoch, (trees, states, data), length=n_steps)

        new_trees, loss = out

        fitness = jnp.min(loss, axis=0)
        trees = jax.vmap(lambda t, i: t[i], in_axes=[1,0])(new_trees, jnp.argmin(loss, axis=0))

        return fitness, trees
    
    def evolve(self, populations, fitness, key):
        populations, left_parents, right_parents, left_children, right_children, reproduction_type, mutate_functions = evolve_populations(self.jit_evolve_population, populations, fitness, key, self.current_generation, self.migration_period, self.migration_size, self.reproduction_type_probabilities, self.reproduction_probabilities, self.tournament_probabilities)
        self.current_generation += 1
        return populations
    
    def mutate_pair(self, parent1, parent2, keys, reproduction_probability):
        offspring, mutate_functions = jax.vmap(self.mutate_trees, in_axes=[0,1,None,None])(jnp.stack([parent1, parent2]), keys, reproduction_probability, self.variable_probabilities)
        return offspring[0], offspring[1], mutate_functions

    def sample_pair(self, parent1, parent2, keys, reproduction_probability):
        offspring = jax.vmap(lambda _keys: jax.vmap(self.sample_tree, in_axes=[0, None, 0])(_keys, self.max_init_depth, self.variable_probabilities), in_axes=[1])(keys)
        return offspring[0], offspring[1], jnp.zeros((2, self.num_trees), dtype=int)