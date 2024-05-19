import warnings
import jax
from jax import Array
import time
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from pathos.multiprocessing import ProcessingPool as Pool
from typing import Tuple, Callable

import genetic_operators.initialization as initialization
from algorithms.strategy_base import Strategy
from networks.tree_policy import TreePolicy

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

class RandomSearch(Strategy):
    """Random search strategy of symbolic expressions.

        Attributes:
            num_generations: The number of generations over which to evolve the population.
            population_size: Number of candidates in the population.
            fitness_function: Function that evaluates a candidate and assigns a fitness.
            current_generation: A reference to keep track of the current generation.
            best_solutions: Best solution at each generation.
            best_fitnesses: Best fitness at each generation.
            expressions: Expressions for each layer in a tree policy.
            layer_sizes: Size of each layer in a tree policy.
            pool: Pool of parallel workers.
            num_populations: Number of subpopulations.
            max_depth: Highest depth of a tree.
            init_method: Method for initializing the trees.
            size_parsinomy: Parsimony factor that increases the fitness of a candidate based on its size.
            leaf_sd: Standard deviation to sample constants.
        """
    def __init__(self, 
                 num_generations: int, 
                 population_size: int,
                 fitness_function: Callable, 
                 expressions: list, 
                 layer_sizes: Array, 
                 num_populations: int = 1, 
                 pool_size: int = 1, 
                 max_depth: int = 7,
                 init_method: str = "grow", 
                 size_parsinomy: float = 1.0, 
                 leaf_sd: float = 1.0):
        super().__init__(num_generations, population_size, fitness_function)
        self.expressions = expressions
        self.layer_sizes = layer_sizes
        self.pool = Pool(pool_size)
        self.pool.close()
        self.num_populations = num_populations
        self.max_depth = max_depth
        self.init_method = init_method
        self.size_parsinomy = size_parsinomy
        self.leaf_sd = leaf_sd

    def initialize_population(self, key: PRNGKey) -> list:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        keys = jrandom.split(key, self.num_populations+1)
        self.pool.restart()
        populations = self.pool.amap(lambda x: initialization.sample_trees(keys[x], self.expressions, self.layer_sizes, self.population_size, self.max_depth, self.init_method, self.leaf_sd), 
                                     range(self.num_populations))
        self.pool.close()
        self.pool.join()
        return populations.get()

    def evaluate_population(self, population: list, data: Tuple) -> Tuple[Array, list]:
        """Evaluates every candidate in population and assigns a fitness.

        :param population: Population of candidates
        :param data: The data required to evaluate the population.

        Returns: Fitness and evaluated population.
        """
        self.pool.restart()
        fitness = self.pool.amap(lambda x: self.evaluate(x, data), [candidate for subpopulation in population for candidate in subpopulation]) #Evaluate each solution parallely on a pool of workers
        self.pool.close()
        self.pool.join()
        tries = 0
        while not fitness.ready():
            time.sleep(1)
            tries += 1
            print(tries)
            if tries >= 2:
                print("TIMEOUT")
                break

        flat_fitnesses = jnp.array(fitness.get())
        fitnesses = jnp.reshape(flat_fitnesses,(self.num_populations, self.population_size))

        #Set the fitness of each solution
        for pop in range(self.num_populations):
            subpopulation = population[pop]
            for candidate in range(self.population_size):
                subpopulation[candidate].set_fitness(fitnesses[pop,candidate] + self.size_parsinomy*len(jax.tree_util.tree_leaves(subpopulation[candidate])))

        best_solution_of_g = find_best_solution(population, fitnesses)
        best_fitness_at_g = self.evaluate(best_solution_of_g, data)

        #Keep track of best solution
        if self.current_generation == 0:
            best_fitness = best_fitness_at_g
            best_solution = best_solution_of_g
        else:
            best_fitness = self.best_fitnesses[self.current_generation - 1]
            best_solution = self.best_solutions[self.current_generation - 1]

            if best_fitness_at_g < best_fitness:
                best_fitness = best_fitness_at_g
                best_solution = best_solution_of_g
            
        self.best_solutions.append(best_solution)
        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)

        return fitnesses, population
    
    def evaluate(self, candidate: TreePolicy, data: Tuple) -> float:
        """Evaluates a candidate and assigns a fitness.

        :param candidate: Candidate solution.
        :param data: The data required to evaluate the population.

        Returns: Fitness of candidate.
        """
        model_functions = candidate.tree_to_function(self.expressions)
        return self.fitness_function(model_functions, data)

    def evolve_population(self, population: list, key: PRNGKey) -> list:   
        """Creates a new population by evolving the current population.

        :param population: Population of candidates
        :param key: Random key.

        Returns: Population with new candidates.
        """        
        keys = jrandom.split(key, self.num_populations)
        self.pool.restart()
        #Evaluate each solution parallely on a pool of workers
        new_populations = self.pool.amap(lambda x: self.update_population(keys[1+x]), range(self.num_populations))
        self.pool.close()
        self.pool.join()

        self.current_generation += 1

        return new_populations.get()
    
    def update_population(self, key) -> list:
        """Samples a new subpopulation.

        :param key: Random key.

        Returns: Population with new candidates.
        """  
        return initialization.sample_trees(key, self.expressions, self.layer_sizes, self.population_size, self.max_depth, self.init_method, self.leaf_sd)