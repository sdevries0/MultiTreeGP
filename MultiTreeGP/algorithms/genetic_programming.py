import warnings

import jax
from jax import Array

import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
import optax
import equinox as eqx
from functools import partial

from typing import Tuple, Callable
import copy
import time

from MultiTreeGP.networks.tree_policy import TreePolicy
from MultiTreeGP.expression import OperatorNode
import MultiTreeGP.genetic_operators.reproduction as reproduction
import MultiTreeGP.genetic_operators.migration as migration
import MultiTreeGP.genetic_operators.initialization as initialization
from MultiTreeGP.algorithms.strategy_base import Strategy

def find_best_solution(populations: list, fitnesses: Array) -> TreePolicy:
    """Returns the best solution in all subpopulations.

        :param populations: Population of candidates
        :param fitnesses: The fitness of all candidates.

        Returns: Best candidate.
        """
    best_solution = None
    best_fitness = jnp.inf
    for pop in range(len(populations)):
        if jnp.min(fitnesses[pop]) < best_fitness:
            best_fitness = jnp.min(fitnesses[pop])
            best_solution = populations[pop][jnp.argmin(fitnesses[pop])]
    return best_solution

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
                 expressions: list, 
                 layer_sizes: Array, 
                 num_populations: int = 1, 
                 max_depth: int = 7, 
                 max_init_depth: int = 4, 
                 tournament_size: int = 7, 
                 init_method: str = "ramped", 
                 size_parsinomy: float = 1.0, 
                 migration_method: str = "ring", 
                 migration_period: int = 5,
                 leaf_sd: float = 1.0, 
                 migration_percentage: float = 0.1,
                 restart_iter_threshold: int = None,
                 gradient_optimisation: bool = False,
                 gradient_steps: int = 10) -> None:
        super().__init__(num_generations, population_size, fitness_function)
        assert len(layer_sizes) == len(expressions), "There is not a set of expressions for every type of layer"
        self.expressions = expressions
        self.layer_sizes = layer_sizes
        self.num_populations = num_populations
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.nr_nodes = 2**self.max_depth-1
        self.init_method = init_method
        self.size_parsinomy = size_parsinomy
        self.leaf_sd = leaf_sd

        self.migration_period = migration_period
        self.migration_method = migration_method
        self.migration_size = int(migration_percentage*population_size)

        self.gradient_optimisation = gradient_optimisation
        self.gradient_steps = gradient_steps

        if restart_iter_threshold == None:
            with warnings.catch_warnings(action="ignore"):
                self.restart_iter_threshold: int = jnp.linspace(10, 5, num_populations, dtype=int)
        else:
            self.restart_iter_threshold = restart_iter_threshold
        self.last_restart = jnp.zeros(num_populations)

        self.tournament_size = tournament_size
        self.selection_pressures = jnp.linspace(0.6,0.9,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(0.85,0.45,self.num_populations),jnp.linspace(0.1,0.5,self.num_populations),
                                         jnp.linspace(0.0,0.0,self.num_populations),jnp.linspace(0.05,0.05,self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(1.0, 1.0, self.num_populations)
        self.elite_percentage = jnp.linspace(0.04, 0.04)

        self.mutation_probabilities = {}
        self.mutation_probabilities["mutate_operator"] = 0.5
        self.mutation_probabilities["delete_operator"] = 0.5
        self.mutation_probabilities["insert_operator"] = 1.0
        self.mutation_probabilities["mutate_constant"] = 0.1
        self.mutation_probabilities["mutate_leaf"] = 0.5
        self.mutation_probabilities["sample_subtree"] = 1.0
        self.mutation_probabilities["prepend_operator"] = 1.0
        self.mutation_probabilities["add_subtree"] = 1.0

        self.best_fitness_per_population = jnp.zeros((num_generations, self.num_populations))

        self.func_dict = {
            "+": -1,
            "-": -2,
            "*": -3,
            "/": -4,
            "**": -5,
            "**2": -6,
            "x0": -7,
            "x1": -8
        }

        # self.func_dict = {
        #     "+": -1,
        #     "*": -2,
        #     "x0": -3,
        #     "x1": -4
        # }

        self.functions = [
            lambda x, y, _data: 0.0,
            lambda x, y, _data: x+y,
            lambda x, y, _data: x-y,
            lambda x, y, _data: x*y,
            lambda x, y, _data: x/y,
            lambda x, y, _data: x**y,
            lambda x, y, _data: x**2,
            lambda x, y, _data: _data[0],
            lambda x, y, _data: _data[1]]

        # self.functions = [
        #     lambda x, y, _data: 0.0,
        #     lambda x, y, _data: x+y,
        #     lambda x, y, _data: x*y,
        #     lambda x, y, _data: _data[0],
        #     lambda x, y, _data: _data[1]]
        
    def initialize_population(self, key: PRNGKey) -> list:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        keys = jrandom.split(key, self.num_populations+1)
        populations = []
        for x in range(self.num_populations):
            population = initialization.sample_trees(keys[x], self.expressions, self.layer_sizes, self.population_size, self.max_init_depth, self.init_method, self.leaf_sd)
            populations.append(population)

        return populations
    
    def map_node(self, tree, matrix, index):
        if isinstance(tree[0], jax.numpy.ndarray):
            matrix = matrix.at[index, 0].set(jnp.abs(tree[0]))
            if tree[0]>0:
                matrix = matrix.at[index, 1].set(1)
            else:
                matrix = matrix.at[index, 1].set(-1)
        else:
            matrix = matrix.at[index, 0].set(self.func_dict[str(tree[0])])

            if len(tree)==3:
                matrix = self.map_node(tree[1], matrix, -self.nr_nodes + 2*index)
                matrix = self.map_node(tree[2], matrix, -self.nr_nodes + 2*index - 1)
                matrix = matrix.at[index, 1].set(-self.nr_nodes + 2*index)
                matrix = matrix.at[index, 2].set(-self.nr_nodes + 2*index - 1)
            if len(tree)==2:
                matrix = self.map_node(tree[1], matrix, -self.nr_nodes + 2*index)
                matrix = matrix.at[index, 1].set(-self.nr_nodes + 2*index)
        return matrix

    def tree_to_matrix(self, candidate):
        matrix = jnp.zeros((jnp.sum(self.layer_sizes), self.nr_nodes, 4))

        trees = candidate()

        for i in range(self.layer_sizes.shape[0]):
            for j in range(self.layer_sizes[i]):
                if i > 0:
                    index = jnp.sum(self.layer_sizes[:i]) + j
                else:
                    index = j
                matrix = matrix.at[index].set(self.map_node(trees[i][j], matrix[index], self.nr_nodes - 1))

        return matrix
    
    def body_fun(self, i, carry):
        array, data = carry
        f_idx, a_idx, b_idx, _ = array.at[i].get()
        
        x = array.at[a_idx.astype(int), 3].get()
        y = array.at[b_idx.astype(int), 3].get()
        value = jax.lax.select(f_idx < 0, jax.lax.switch(-f_idx.astype(int), self.functions, x, y, data), f_idx * a_idx)
        
        array = array.at[i, 3].set(value)

        return (array, data)

    def foriloop(self, data, array):
        x, _ = jax.lax.fori_loop(0, self.nr_nodes, self.body_fun, (array, data))
        return data, x.at[-1, -1].get()
    
    # def vmap_foriloop(self, array, data):
    #     result = jax.vmap(self.foriloop, in_axes=[0, None])(array, data)
    #     return result.at[:, -1, -1].get()

    def vmap_foriloop(self, array, data):
        _, result = jax.lax.scan(self.foriloop, init=data, xs=array)
        return jnp.array(result)

    def evaluate_population(self, populations: list, data: Tuple) -> Tuple[Array, list]:
        """Evaluates every candidate in population and assigns a fitness.

        :param population: Population of candidates
        :param data: The data required to evaluate the population.

        Returns: Fitness and evaluated population.
        """
      
        population_matrix = jnp.zeros((self.num_populations*self.population_size, jnp.sum(self.layer_sizes), self.nr_nodes, 4))

        start = time.time()
        for pop, population in enumerate(populations):
            for i in range(self.population_size):
                population_matrix = population_matrix.at[i + pop*self.population_size].set(self.tree_to_matrix(population[i]))
        print("map", time.time()-start)

        start = time.time()

        fitness = jax.vmap(self.fitness_function, in_axes=[0, None, None])(population_matrix, jax.jit(self.vmap_foriloop), data)

        print("eval", time.time() - start)

        fitnesses = fitness.reshape((self.num_populations, self.population_size))

        for pop, population in enumerate(populations):
            for i in range(self.population_size):
                population[i].set_fitness(fitnesses[pop, i] + self.size_parsinomy*len(jax.tree_util.tree_leaves(population[i]())))

        self.best_fitness_per_population = self.best_fitness_per_population.at[self.current_generation].set(jnp.min(fitnesses, axis=1))
        best_solution_of_g = find_best_solution(populations, fitnesses)
        best_fitness_of_g = best_solution_of_g.fitness

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
            
        self.best_solutions.append(best_solution)
        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)

        return fitness, populations
    
    # def replace_ones(self, tree):
    #     if len(tree) == 3:
    #         operator = tree[0]
    #         left_tree, left_bool = self.replace_ones(tree[1])
    #         right_tree, right_bool = self.replace_ones(tree[2])
    #         if operator.string == "+" or operator.string == "-":
    #             if left_bool and right_bool:
    #                 return [tree[0], left_tree, right_tree], True
    #             elif left_bool:
    #                 return [tree[0], left_tree, [OperatorNode(lambda x, y: x * y, "*", 2), [jnp.array(1)], right_tree]], True
    #             elif right_bool:
    #                 return [tree[0], [OperatorNode(lambda x, y: x * y, "*", 2), [jnp.array(1)], left_tree], right_tree], True
    #             else:
    #                 return [tree[0], [OperatorNode(lambda x, y: x * y, "*", 2), [jnp.array(1)], left_tree], [OperatorNode(lambda x, y: x * y, "*", 2), [jnp.array(1)], right_tree]], True
    #         else:
    #             if left_bool or right_bool:
    #                 return [tree[0], left_tree, right_tree], True
    #             else:
    #                 return [tree[0], left_tree, right_tree], False
    #     if len(tree) ==1:
    #         if isinstance(tree[0], jax.numpy.ndarray):
    #             return tree, True
    #         else:
    #             return tree, False
    
    # def evaluate(self, candidate: TreePolicy, data: Tuple, optimise: bool = False, num_steps: int = 3) -> TreePolicy:
    #     """Evaluates a candidate and assigns a fitness and optionally optimises the constants in the tree.

    #     :param candidate: Candidate solution.
    #     :param data: The data required to evaluate the population.
    #     :param optimise: Whether to optimise the constants in the tree.
    #     :param num_steps: Number of steps during constant optimisation.

    #     Returns: Optimised and evaluated candidate.
    #     """
    #     def optimise_tree(tree, paths):
    #         params, paths = tree.get_params()
    #         optim = optax.adam(learning_rate=0.05, b1=0.6, b2=0.85)
    #         state = optim.init(params)

    #         def loss_f(params, data):
    #             policy = copy.copy(tree)
    #             policy.set_params(jnp.array(params), paths)
    #             f = policy.tree_to_function(self.expressions)
                
    #             return self.fitness_function(f, data)

    #         def epoch(_, carry):
    #             _, params, state = carry
    #             loss = loss_f(params, data)
    #             gradient = jax.grad(loss_f)(params, data)
    #             updates, state = optim.update(gradient, state, params)
    #             params = jax.tree_util.tree_map(lambda x,y: x+y, params, updates)
    #             return loss, params, state

    #         loss, params, _ = jax.lax.fori_loop(0, num_steps, epoch, (jnp.array(0), params, state))

    #         # for i in range(num_steps):
    #         #     loss = loss_f(params, data)
    #         #     print(loss)
    #         #     gradient = jax.grad(loss_f)(params, data)
    #         #     updates, state = optim.update(gradient, state, params)
    #         #     params = jax.tree_util.tree_map(lambda x,y: x+y, params, updates)

    #         final_tree = copy.copy(tree)
    #         final_tree.set_params(jnp.array(params), paths)

    #         return loss, final_tree
        
    #     fitness = self.fitness_function(candidate.tree_to_function(self.expressions), data)

    #     if optimise and fitness < 500:
    #         # _candidate, _ = self.replace_ones(candidate())
    #         # candidate = eqx.tree_at(lambda t: t()[0][0], candidate, _candidate)
    #         params, paths = candidate.get_params()
    #         if len(params)>0:
    #             loss, new_candidate = optimise_tree(candidate, paths)

    #             if loss < fitness:
    #                 candidate = new_candidate
    #                 fitness = loss

    #     candidate.set_fitness(fitness)
    #     return candidate

    def evolve_population(self, population: list, key: PRNGKey) -> list:
        """Creates a new population by evolving the current population.

        :param population: Population of candidates
        :param key: Random key.

        Returns: Population with new candidates.
        """        

        #Migrate individuals between populations every few generations
        if ((self.current_generation+1)%self.migration_period)==0:
            key, new_key = jrandom.split(key)
            population = migration.migrate_populations(population, self.migration_method, self.migration_size, new_key)

        #Check if population has not improved during last generations.
        self.last_restart = self.last_restart + 1
        restart = jnp.logical_and(self.last_restart>self.restart_iter_threshold, self.best_fitness_per_population[self.current_generation] >= 
                                  jnp.array([self.best_fitness_per_population[self.current_generation-self.restart_iter_threshold[i], i] for i in range(self.num_populations)]))
        self.last_restart = self.last_restart.at[restart].set(0)
        self.current_generation += 1
            
        keys = jrandom.split(key, self.num_populations+1)

        new_populations = []
        for x in range(self.num_populations):
            new_population = self.update_population(x, population[x], restart[x], keys[x])
            new_populations.append(new_population)
        return new_populations
    
    def update_population(self, index: int, population: list, restart: bool, key: PRNGKey) -> list:
        """Either samples or evolves a new subpopulation.

        :param index: Index of current subpopulation.
        :param population: Current subpopulation.
        :param restart: Indicates whether the population should be restarted.
        :param key: Random key.

        Returns: Population with new candidates.
        """  
        if restart:
            print("restart in", index)
            return initialization.sample_trees(key, self.expressions, self.layer_sizes, self.population_size, self.max_init_depth, self.init_method, self.leaf_sd)
        else:
            return reproduction.next_population(population, key, self.expressions, self.layer_sizes, self.reproduction_type_probabilities[index], self.reproduction_probabilities[index], 
                                        self.mutation_probabilities, self.tournament_probabilities[index], self.tournament_size, self.max_depth, self.max_init_depth, self.elite_percentage[index], self.leaf_sd)