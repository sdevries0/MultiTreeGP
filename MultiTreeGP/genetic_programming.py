import jax
print("These device(s) are detected: ", jax.devices())
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

from MultiTreeGP.genetic_operators.crossover import crossover_trees
from MultiTreeGP.genetic_operators.initialization import sample_population, sample_tree
from MultiTreeGP.genetic_operators.mutation import initialize_mutation_functions
from MultiTreeGP.genetic_operators.reproduction import evolve_populations, evolve_population

#Function containers
def lambda_operator_arity1(f):
    return lambda x, y, _data: f(x)

def lambda_operator_arity2(f):
    return lambda x, y, _data: f(x, y)

def lambda_leaf(i):
    return lambda x, y, _data: _data[i]

class GeneticProgramming:
    """Genetic programming strategy of symbolic expressions.

        :param num_generations: The number of generations over which to evolve the population
        :param population_size: Number of candidates in the population
        :param fitness_function: Function that evaluates a candidate and assigns a fitness
        :param operator_list: List of operators that can be included in the trees
        :param variable_list: List of variables that can be included in the trees, which can vary between layers of a candidate
        :param layer_sizes: Size of each layer in a candidate
        :param num_populations: Number of subpopulations
        :param max_init_depth: Highest depth of a tree during initialization
        :param max_nodes: Maximum number of nodes in a tree
        :param device_type: Type of device on which the evaluation and evolution takes place
        :param tournament_size: Size of the tournament
        :param size_parsinomy: Parsimony factor that increases the fitness of a candidate based on its size
        :param coefficient_sd: Standard deviation to sample coefficients
        :param migration_period: Number of generations after which populations are migrated
        :param migration_percentage: Number of candidates to migrate
        :param elite_percentage: Percentage of elite candidates that procceed to the next population
        :param coefficient_optimisation: If the coefficients are optimised with gradients
        :param gradient_steps: For how many steps the coefficients are optimised
        :param optimiser: Optimiser for coefficient optimisation
        :param selection_pressure_factors: The selection pressure for each subpopulation
        :param reproduction_probability_factors: The reproduction probability for each subpopulation
        :param crossover_probability_factors: The crossover probability for each subpopulation
        :param mutation_probability_factors: The mutation probability for each subpopulation
        :param sample_probability_factor: The probability to sample a new candidate for each subpopulation
        """
    def __init__(self, num_generations: int, 
                 population_size: int, 
                 fitness_function: Callable, 
                 operator_list: list,
                 variable_list: list,
                 layer_sizes: Array,
                 num_populations: int = 1,
                 max_init_depth: int = 4, 
                 max_nodes: int = 30,
                 device_type: str = 'cpu',
                 tournament_size: int = 7, 
                 size_parsinomy: float = 1.0, 
                 coefficient_sd: float = 1.0,
                 migration_period: int = 10,
                 migration_percentage: float = 0.1,
                 elite_percentage: int = 0.1,
                 coefficient_optimisation: bool = False,
                 gradient_steps: int = 10,
                 optimiser = optax.adam(learning_rate=0.001, b1=0.9, b2=0.999),
                 selection_pressure_factors: Tuple[float] = (0.6, 0.9),
                 reproduction_probability_factors: Tuple[float] = (1.0, 0.5),
                 crossover_probability_factors: Tuple[float] = (0.9, 0.4),
                 mutation_probability_factors: Tuple[float] = (0.1, 0.5),
                 sample_probability_factors: Tuple[float] = (0.0, 0.1)) -> None:
        
        self.layer_sizes = layer_sizes
        assert num_populations>0, "The number of populations should be larger than 0"
        self.num_populations = num_populations
        assert population_size>0 and population_size%2==0, "The population_size should be larger than 0 and an even number"
        self.population_size = population_size
        assert max_init_depth>0, "The max initial depth should be larger than 0"
        self.max_init_depth = max_init_depth
        assert max_nodes>0, "The max number of nodes should be larger than 0"
        self.max_nodes = max_nodes
        self.num_trees = jnp.sum(self.layer_sizes)
        assert self.num_trees>0, "The number of trees should be larger than 0"

        self.current_generation = 0
        assert num_generations>0, "The number of generations should be larger than 0"
        self.best_fitnesses = jnp.zeros(num_generations)
        self.best_solutions = jnp.zeros((num_generations, self.num_trees, self.max_nodes, 4))

        self.size_parsinomy = size_parsinomy
        self.coefficient_sd = coefficient_sd

        assert migration_period>1, "The migration period should be larger than 1"
        self.migration_period = migration_period
        assert migration_percentage*population_size%1==0, "The migration size should be an integer"
        self.migration_size = int(migration_percentage*population_size)

        assert tournament_size>1, "The number of gradient steps should be larger than 1"
        self.tournament_size = tournament_size
        self.selection_pressures = jnp.linspace(*selection_pressure_factors, self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(*crossover_probability_factors, self.num_populations),
                                                           jnp.linspace(*mutation_probability_factors, self.num_populations),
                                                           jnp.linspace(*sample_probability_factors, self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(*reproduction_probability_factors, self.num_populations)

        self.elite_size = int(elite_percentage*population_size)
        assert self.elite_size%2==0, "The elite size should be a multiple of two"

        self.coefficient_optimisation = coefficient_optimisation
        if coefficient_optimisation:
            assert gradient_steps>0, "The number of gradient steps should be larger than 0"
        self.gradient_steps = gradient_steps
        self.optimiser = optimiser

        self.map_b_to_d = self.create_map_b_to_d(self.max_init_depth)

        #Initialize library of nodes
        string_to_node = {} #Maps string to node
        node_to_string = {}
        node_function_list = [lambda x, y, _data: 0.0, lambda x, y, _data: 0.0]

        n_operands = [0, 0]
        index = 2
        operator_probabilities = jnp.zeros(len(operator_list))

        assert len(operator_list)>0, "No operators were given"

        for operator_tuple in operator_list:
            string = operator_tuple[0]
            f = operator_tuple[1]
            arity = operator_tuple[2]
            if len(operator_tuple)==4:
                probability = operator_tuple[3]
            else:
                probability = 1.0

            if string not in string_to_node:
                string_to_node[string] = index
                node_to_string[index] = string
                if arity==1:
                    node_function_list.append(lambda_operator_arity1(f))
                    n_operands.append(1)
                elif arity==2:
                    node_function_list.append(lambda_operator_arity2(f))
                    n_operands.append(2)
                operator_probabilities = operator_probabilities.at[index-2].set(probability)
                index += 1

        self.operator_probabilities = operator_probabilities
        self.operator_indices = jnp.arange(2, index)
        var_start_index = index

        data_index = 0
        assert len(layer_sizes) == len(variable_list), "There is not a set of expressions for every type of layer"

        for var_list in variable_list:
            assert len(var_list)>0, "An empty set of variables was given"
            for var in var_list:
                if var not in string_to_node:
                    string_to_node[var] = index
                    node_to_string[index] = var
                    node_function_list.append(lambda_leaf(data_index))
                    n_operands.append(0)
                    index += 1
                    data_index += 1
        
        self.variable_indices = jnp.arange(var_start_index, index)
        variable_array = jnp.zeros((self.num_trees, data_index))

        counter = 0
        for layer_i, var_list in enumerate(variable_list):
            p = jnp.zeros((data_index))
            for var in var_list:
                p = p.at[string_to_node[var] - var_start_index].set(1)

            for _ in range(layer_sizes[layer_i]):
                variable_array = variable_array.at[counter].set(p)
                counter += 1

        self.slots = jnp.array(n_operands)
        self.string_to_node = string_to_node
        self.node_to_string = node_to_string
        self.node_function_list = node_function_list
        self.variable_array = variable_array

        print(f"Input data should be formatted as: {[self.node_to_string[i.item()] for i in self.variable_indices]}.")

        #Define jittable reproduction functions
        self.sample_args = (self.variable_indices, 
                            self.operator_indices, 
                            self.operator_probabilities, 
                            self.slots, 
                            self.coefficient_sd, 
                            self.map_b_to_d)
        
        self.sample_population = partial(sample_population, 
                                         num_trees = self.num_trees, 
                                         max_init_depth = self.max_init_depth, 
                                         max_nodes = self.max_nodes, 
                                         variable_array = self.variable_array,
                                         args = self.sample_args)
        
        self.sample_tree = partial(sample_tree, 
                                   max_nodes = self.max_nodes, 
                                   max_init_depth = self.max_init_depth, 
                                   args = self.sample_args)

        self.mutate_args = (self.sample_tree, 
                            self.max_nodes, 
                            self.max_init_depth, 
                            self.variable_indices, 
                            self.operator_indices, 
                            self.operator_probabilities, 
                            self.slots, 
                            self.coefficient_sd)
        
        self.mutate_trees = initialize_mutation_functions(self.mutate_args)

        self.partial_crossover = partial(crossover_trees, 
                                         operator_indices = self.operator_indices, 
                                         max_nodes = self.max_nodes)

        self.reproduction_functions = [self.partial_crossover, self.mutate_pair, self.sample_pair]

        self.jit_evolve_population = jax.jit(partial(evolve_population, 
                                                     reproduction_functions = self.reproduction_functions, 
                                                     elite_size = self.elite_size, 
                                                     tournament_size = self.tournament_size, 
                                                     num_trees = self.num_trees, 
                                                     population_size=population_size))

        #Define partial fitness function for evaluation
        self.jit_body_fun = jax.jit(partial(self.body_fun, node_function_list = self.node_function_list))
        self.partial_ff = partial(fitness_function, tree_evaluator = self.vmap_foriloop)

        #Define parallel evaluation functions
        self.vmap_trees = jax.vmap(self.partial_ff, in_axes=[0, 0, None])
        self.vmap_gradients = jax.vmap(jax.value_and_grad(self.partial_ff), in_axes=[0, 0, None])

        devices = mesh_utils.create_device_mesh((len(jax.devices(device_type))))
        self.mesh = Mesh(devices, axis_names=('i'))

        #Define sharded functions for evaluation and optimisation
        @partial(shard_map, mesh=self.mesh, in_specs=(P('i'), P(None)), out_specs=P('i'), check_rep=False)
        def shard_eval(array, data):
            result = self.vmap_trees(array[...,3:], array[...,:3], data)
            return result
            
        @partial(shard_map, mesh=self.mesh, in_specs=(P('i'), P(None)), out_specs=(P('i'), P('i')), check_rep=False)
        def shard_optimise(array, data):
            result, _array = self.optimise(array, data, self.gradient_steps)
            return result, _array
        
        self.jit_eval = jax.jit(shard_eval)
        self.jit_optimise = jax.jit(shard_optimise)

    def create_map_b_to_d(self, depth: int) -> Array:
        """
        Creates a mapping from the breadth first index to depth first index given a depth

        :param depth

        Returns: Index mapping
        """
                
        max_nodes = 2**depth-1
        current_depth = 0
        map_b_to_d = jnp.zeros(max_nodes)
        
        for i in range(max_nodes):
            if i>0:
                parent = (i + (i%2) - 2)//2 #Determine parent position
                value = map_b_to_d[parent]
                if (i % 2)==0: #Right child
                    new_value = value + 2**(depth-current_depth+1) 
                else: #Left child
                    new_value = value + 1
                map_b_to_d = map_b_to_d.at[i].set(new_value)
            current_depth += i==(2**current_depth-1) #If last node at current depth is reached, increase current depth

        return max_nodes - 1 - map_b_to_d #Inverse the mapping

    def initialize_population(self, key: PRNGKey) -> Array:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        keys = jr.split(key, self.num_populations)
        populations = jax.vmap(self.sample_population, in_axes=[0, None])(keys, self.population_size)

        return populations
    
    def tree_to_string(self, tree: Array) -> str:
        """
        Maps tree to string

        :param tree

        Returns: String representation of tree
        """
        if tree[-1,0]==1: #Coefficient
            return "{:.2f}".format(tree[-1,3])
        elif tree[-1,1]<0: #Variable
            return self.node_to_string[tree[-1,0].astype(int).item()]
        elif tree[-1,2]<0: #Operator with one operand
            substring = self.tree_to_string(tree[:tree[-1,1].astype(int)+1])
            return f"{self.node_to_string[tree[-1,0].astype(int).item()]}({substring})"
        else: #Operator with two operands
            substring1 = self.tree_to_string(tree[:tree[-1,1].astype(int)+1])
            substring2 = self.tree_to_string(tree[:tree[-1,2].astype(int)+1])
            return f"({substring1}){self.node_to_string[tree[-1,0].astype(int).item()]}({substring2})"
        
    def to_string(self, candidate: Array) -> str:
        """
        Maps trees in a candidate to string
        
        :param candidate

        Returns: String representation of candidate
        """
        string_output = ""
        tree_index = 0
        layer_index = 0
        for tree in candidate:
            if tree_index==0: #Begin layer of trees
                string_output += "["
            string_output += str(sympy.parsing.sympy_parser.parse_expr(self.tree_to_string(tree))) #Map tree to string
            if tree_index < (self.layer_sizes[layer_index] - 1): #Continue layer of trees
                string_output += ", "
                tree_index += 1
            else: #End layer of trees
                string_output += "]"
                if layer_index < (self.layer_sizes.shape[0] - 1): #Begin new layer
                    string_output += ", "
                tree_index = 0
                layer_index += 1
        return string_output
    
    def body_fun(self, i, carry: Tuple[Array, Array], node_function_list):
        """
        Evaluates a node given inputs
        
        :param tree
        :param data
        :param node_function_list: Maps nodes to callable functions

        Returns: Evaluated node
        """

        tree, data = carry
        f_idx, a_idx, b_idx, coefficient = tree[i] #Get node function, index of first and second operand, and coefficient value of node (which will be 0 if the node function is not 1)
    
        x = tree[a_idx.astype(int), 3] #Value of first operand
        y = tree[b_idx.astype(int), 3] #Value of second operand
        value = jax.lax.select(f_idx == 1, coefficient, jax.lax.switch(f_idx.astype(int), node_function_list, x, y, data)) #Computes value of the node
        
        tree = tree.at[i, 3].set(value) #Store value

        return (tree, data)

    def foriloop(self, tree: Array, data: Array) -> Array:
        """
        Loops through a tree to compute the value of each node bottom up 
        
        :param tree
        :param data

        Returns: Value of the root node
        """
        x, _ = jax.lax.fori_loop(0, self.max_nodes, self.jit_body_fun, (tree, data))
        return x[-1, -1]

    def vmap_foriloop(self, candidate: Array, data: Array) -> Array:
        """
        Calls the evaluation function for each tree in a candidate

        :param candidate
        :param data

        Returns: Result of each tree
        """

        result = jax.vmap(self.foriloop, in_axes=[0, None])(candidate, data)
        return result
       
    def evaluate_population(self, populations: Array, data: Tuple) -> Tuple[Array, Array]:
        """Evaluates every candidate in population and assigns a fitness. Optionally the coefficients in the candidates are optimised

        :param population: Population of candidates
        :param data: The data required to evaluate the population.

        Returns: Fitness and evaluated or optimised population.
        """

        flat_populations = populations.reshape(self.num_populations*self.population_size, *populations.shape[2:]) #Flatten the populations so they can be distributed over the devices
        flat_populations = jax.device_put(flat_populations, NamedSharding(self.mesh, P('i')))
        
        fitness = self.jit_eval(flat_populations, data) #Evaluate the candidates

        #Optimise coefficients of the best candidates given conditions
        if (self.coefficient_optimisation & (self.current_generation>10) & ((self.current_generation+1)%5==0)):
            best_candidates_idx = jnp.argsort(fitness)[:50]
            optimised_fitness, optimised_population = self.jit_optimise(flat_populations[best_candidates_idx], data)
            flat_populations = flat_populations.at[best_candidates_idx].set(optimised_population)
            fitness = fitness.at[best_candidates_idx].set(optimised_fitness)

        fitness = fitness + jax.vmap(lambda array: self.size_parsinomy * jnp.sum(array[:,:,0]!=0))(flat_populations) #Increase fitness based on the size of the candidate

        best_solution = flat_populations[jnp.argmin(fitness)]
        best_fitness = jnp.min(fitness)
            
        #Store best fitness and solution
        self.best_solutions = self.best_solutions.at[self.current_generation].set(best_solution)
        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)

        return fitness.reshape((self.num_populations, self.population_size)), flat_populations.reshape((self.num_populations, self.population_size, *flat_populations.shape[1:]))      
            
    def epoch(self, carry: Tuple[Array, Array, Tuple], x: int) -> Tuple[Tuple[Array, Array, Tuple], Tuple[Array, Array]]:
        """
        Applies one step of coefficient optimisation to a batch of candidates

        :param candidates
        :param states: Optimiser states of each candidate
        :param data
        
        Returns: Candidates with optimised coefficients
        """

        candidates, states, data = carry
        loss, gradients = self.vmap_gradients(candidates[...,3:], candidates[...,:3], data) #Compute loss and gradients parallely

        updates, states = jax.vmap(self.optimiser.update)(gradients, states, candidates[...,3]) #Compute updates parallely
        new_candidates = candidates.at[...,3:].set(jax.vmap(lambda t, u: t + u)(candidates[...,3:], updates)) #Apply updates to coefficients parallely
        
        return (new_candidates, states, data), (candidates, loss)

    def optimise(self, candidates: Array, data: Tuple, n_epoch: int):
        """Optimises the constants in the candidates

        :param candidates: Candidate solutions
        :param data: The data required to evaluate the population
        :param n_epoch: Number of steps to optimise coefficients

        Returns: Optimised and evaluated candidate.
        """

        states = jax.vmap(self.optimiser.init)(candidates[...,3:]) #Initialize optimisers for each candidate

        _, out = jax.lax.scan(self.epoch, (candidates, states, data), length=n_epoch)

        new_candidates, loss = out

        fitness = jnp.min(loss, axis=0) #Get best fitness during optimisation
        candidates = jax.vmap(lambda t, i: t[i], in_axes=[1,0])(new_candidates, jnp.argmin(loss, axis=0)) #Get best candidate during optimisation

        return fitness, candidates
    
    def evolve(self, populations: Array, fitness: Array, key: PRNGKey) -> Array:
        """
        Evolves each population independently

        :param population: Populations of candidates
        :param fitness: Fitness of candidates
        :param key

        Returns: Evolved populations
        
        """
        populations = evolve_populations(self.jit_evolve_population, 
                                         populations, 
                                         fitness, 
                                         key, 
                                         self.current_generation, 
                                         self.migration_period, 
                                         self.migration_size, 
                                         self.reproduction_type_probabilities, 
                                         self.reproduction_probabilities, 
                                         self.tournament_probabilities)
        self.current_generation += 1
        return populations
    
    def mutate_pair(self, parent1: Array, parent2: Array, keys: Array, reproduction_probability: float) -> Tuple[Array, Array]:
        """
        Mutates a pair of candidates

        :param parent1
        :param parent2
        :param keys
        :param reproduction_probability: Probability of a tree to be mutated
        
        Returns: Pair of candidates after mutation
        """
        offspring = jax.vmap(self.mutate_trees, in_axes=[0,1,None,None])(jnp.stack([parent1, parent2]), keys, reproduction_probability, self.variable_array)
        return offspring[0], offspring[1]

    def sample_pair(self, parent1: Array, parent2: Array, keys: Array, reproduction_probability: float) -> Tuple[Array, Array]:
        """
        Samples a pair of candidates

        :param parent1
        :param parent2
        :param keys
        :param reproduction_probability: Probability of a tree to be mutated
        
        Returns: Pair of candidates 
        """
        offspring = jax.vmap(lambda _keys: jax.vmap(self.sample_tree, in_axes=[0, None, 0])(_keys, self.max_init_depth, self.variable_array), in_axes=[1])(keys)
        return offspring[0], offspring[1]
    
    def get_statistics(self, generation: int = None) -> Tuple[Array | int, Array]:
        """Returns best fitness and best solution.

        :param generation: Generation of which the best fitness and solution are required. If None, returns all best fitness and solutions.

        Returns: Best fitness and best solution.
        """
        if generation is not None:
            return self.best_fitnesses[generation], self.best_solutions[generation]
        else:
            return self.best_fitnesses, self.best_solutions