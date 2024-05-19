import warnings
import jax
from jax import Array
import time
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from pathos.multiprocessing import ProcessingPool as Pool
from typing import Tuple, Callable
import copy
from scipy.optimize import minimize
import equinox as eqx
from networks.tree_policy import TreePolicy

import genetic_operators.reproduction as reproduction
import genetic_operators.migration as migration
import genetic_operators.initialization as initialization
from algorithms.strategy_base import Strategy

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
            pool: Pool of parallel workers.
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
                 pool_size: int = 1,
                 max_depth: int = 7, 
                 max_init_depth: int = 4, 
                 tournament_size: int = 7, 
                 init_method: str = "ramped", 
                 size_parsinomy: float = 1.0, 
                 migration_method: str = "ring", 
                 migration_period: int = 5, 
                 leaf_sd: float = 1.0, 
                 migration_percentage: float = 0.1, 
                 restart_iter_threshold: int = None) -> None:
        super().__init__(num_generations, population_size, fitness_function)
        self.expressions = expressions
        self.layer_sizes = layer_sizes
        self.pool = Pool(pool_size)
        self.pool.close()
        self.num_populations = num_populations
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.init_method = init_method
        self.size_parsinomy = size_parsinomy
        self.leaf_sd = leaf_sd

        self.migration_period = migration_period
        self.migration_method = migration_method
        self.migration_size = int(migration_percentage*population_size)

        if restart_iter_threshold == None:
            with warnings.catch_warnings(action="ignore"):
                self.restart_iter_threshold: int = jnp.linspace(10, 4, num_populations, dtype=int)
        else:
            self.restart_iter_threshold = restart_iter_threshold
        self.last_restart = jnp.zeros(num_populations)

        self.tournament_size = tournament_size
        self.selection_pressures = jnp.linspace(0.4,1.0,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(0.65,0.3,self.num_populations),jnp.linspace(0.3,0.3,self.num_populations),
                                         jnp.linspace(0.0,0.4,self.num_populations),jnp.linspace(0.05,0.0,self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(0.4, 1.0, self.num_populations)
        self.elite_percentage = jnp.linspace(0.05, 0)

        self.mutation_probabilities = {}
        self.mutation_probabilities["mutate_operator"] = 0.5
        self.mutation_probabilities["delete_operator"] = 1.0
        self.mutation_probabilities["insert_operator"] = 0.5
        self.mutation_probabilities["mutate_constant"] = 0.5
        self.mutation_probabilities["mutate_leaf"] = 0.5
        self.mutation_probabilities["sample_subtree"] = 1.0
        self.mutation_probabilities["prepend_operator"] = 0.5
        self.mutation_probabilities["add_subtree"] = 1.0

        self.best_fitness_per_population = jnp.zeros((num_generations, self.num_populations))

    def initialize_population(self, key: PRNGKey) -> list:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        keys = jrandom.split(key, self.num_populations+1)
        self.pool.restart()
        populations = self.pool.amap(lambda x: initialization.sample_trees(keys[x], self.expressions, self.layer_sizes, self.population_size, self.max_init_depth, self.init_method, self.leaf_sd), 
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
        # population_pool = self.pool.amap(lambda x: self.evaluate(x, data) if x.fitness == None else x.fitness, [candidate for subpopulation in population for candidate in subpopulation]) #Evaluate each solution parallely on a pool of workers
        population_pool = self.pool.amap(lambda x: self.evaluate(x, data, optimise=(self.current_generation>15) & (self.current_generation%3==0), num_steps=3), [candidate for subpopulation in population for candidate in subpopulation]) #Evaluate each solution parallely on a pool of workers
        
        self.pool.close()
        self.pool.join()
        tries = 0
        while not population_pool.ready():
            time.sleep(1)
            tries += 1
            print(tries)
            if tries >= 2:
                print("TIMEOUT")
                break

        flat_population = population_pool.get()
        
        optimised_population = []
        for pop in range(self.num_populations):
            optimised_population.append(flat_population[pop * self.population_size : (pop+1) * self.population_size])

        #Set the fitness of each solution
        fitnesses = jnp.zeros((self.num_populations, self.population_size))
        for pop in range(self.num_populations):
            for candidate in range(self.population_size):
                fitnesses = fitnesses.at[pop, candidate].set(optimised_population[pop][candidate].fitness + self.size_parsinomy*len(jax.tree_util.tree_leaves(optimised_population[pop][candidate]())))

        self.best_fitness_per_population = self.best_fitness_per_population.at[self.current_generation].set(jnp.min(fitnesses, axis=1))
        best_solution_of_g = find_best_solution(optimised_population, fitnesses)
        best_fitness_of_g = self.evaluate(best_solution_of_g, data, self.expressions, optimise=False).fitness

        #Keep track of best solution
        if self.current_generation == 0:
            best_fitness = best_fitness_of_g
            best_solution = best_solution_of_g
        else:
            best_fitness = self.best_fitnesses[self.current_generation - 1]
            best_solution = self.best_solutions[self.current_generation - 1]

            # best_fitness = self.evaluate(best_solution, data, self.expressions, optimise=False).fitness

            if best_fitness_of_g < best_fitness:
                best_fitness = best_fitness_of_g
                best_solution = best_solution_of_g
            
        self.best_solutions.append(best_solution)
        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)

        return fitnesses, optimised_population
    
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
    
    def evaluate(self, candidate: TreePolicy, data: Tuple, optimise: bool = False, num_steps: int = 3) -> TreePolicy:
        """Evaluates a candidate and assigns a fitness and optionally optimises the constants in the tree.

        :param candidate: Candidate solution.
        :param data: The data required to evaluate the population.
        :param optimise: Whether to optimise the constants in the tree.
        :param num_steps: Number of steps during constant optimisation.

        Returns: Optimised and evaluated candidate.
        """
        def optimize(model, fitness_function, data, paths, params):
            candidate = copy.copy(model)
            candidate.set_params(jnp.array(params), paths)
            return fitness_function(candidate.tree_to_function(self.expressions), data)

        if optimise:
            # model2, _ = self.replace_ones(model()[0][0])
            # model = eqx.tree_at(lambda t: t()[0][0], model, model2)
            params, paths = candidate.get_params()
            if len(params)==0:
                fitness = self.fitness_function(candidate.tree_to_function(self.expressions), data)
                candidate.set_fitness(fitness)
            elif candidate.n_optimisation_steps > 4:
                fitness = self.fitness_function(candidate.tree_to_function(self.expressions), data)
                candidate.set_fitness(fitness)
            else:
                best_params = minimize(lambda p: optimize(candidate, self.fitness_function, data, paths, p), params, tol=1e-1, method='Nelder-Mead', options={'maxiter':num_steps})
                candidate.set_params(jnp.array(best_params.x), paths)
                candidate.set_fitness(jnp.array(best_params.fun))
                candidate.update_steps(num_steps)
        else:
            fitness = self.fitness_function(candidate.tree_to_function(self.expressions), data)
            candidate.set_fitness(fitness)

        return candidate

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
        self.pool.restart()
        #Evaluate each solution parallely on a pool of workers
        new_populations = self.pool.amap(lambda x: self.update_population(x, population[x], restart[x], keys[x]), range(self.num_populations))
        self.pool.close()
        self.pool.join()
        return new_populations.get()
    
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
