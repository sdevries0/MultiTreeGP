import warnings
import jax
import time
import jax.numpy as jnp
import jax.random as jrandom
from pathos.multiprocessing import ProcessingPool as Pool

import genetic_operators.reproduction as reproduction
import miscellaneous.helper_functions as helper_functions
import genetic_operators.migration as migration
import genetic_operators.initialization as initialization
from algorithms.strategy_base import Strategy
from genetic_operators import simplification

class RandomSearch(Strategy):
    def __init__(self, num_generations, population_size, fitness_function, expressions, layer_sizes, num_populations = 1, state_size = 1, pool_size = 1, max_depth = 7
                 , init_method = "grow", size_parsinomy = 1):
        super().__init__(num_generations, population_size, fitness_function)
        self.expressions = expressions
        self.layer_sizes = layer_sizes
        self.pool = Pool(pool_size)
        self.pool.close()
        self.num_populations = num_populations
        self.state_size = state_size
        self.max_depth = max_depth
        self.init_method = init_method
        self.size_parsinomy = size_parsinomy

    def initialize_population(self, key):
        #Sample a specified number of trees
        keys = jrandom.split(key, self.num_populations+1)
        self.pool.restart()
        populations = self.pool.amap(lambda x: initialization.sample_trees(keys[x], self.expressions, self.layer_sizes, self.population_size, self.max_depth, self.init_method), range(self.num_populations))
        self.pool.close()
        self.pool.join()
        return populations.get()

    def evaluate_population(self, population, data):
        self.pool.restart()
        fitness = self.pool.amap(lambda x: self.evaluate(x, data), helper_functions.flatten(population)) #Evaluate each solution parallely on a pool of workers
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

        best_solution_of_g = helper_functions.best_solution(population, fitnesses)
        best_fitness_at_g = self.evaluate(best_solution_of_g, data)

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

        self.current_generation += 1

        return fitnesses, population
    
    def evaluate(self, model, data):
        model_functions = model.tree_to_function(self.expressions)
        return self.fitness_function(model_functions, data)

    def sample_population(self, key, population):           
        keys = jrandom.split(key, self.num_populations)
        self.pool.restart()
        #Evaluate each solution parallely on a pool of workers
        new_populations = self.pool.amap(lambda x: self.update_population(keys[1+x]), range(self.num_populations))
        self.pool.close()
        self.pool.join()
        return new_populations.get()
    
    def update_population(self, key):
        return initialization.sample_trees(key, self.expressions, self.layer_sizes, self.population_size, self.max_depth, self.init_method)