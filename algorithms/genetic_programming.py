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

class GeneticProgramming(Strategy):
    def __init__(self, num_generations, population_size, fitness_function, expressions, layer_sizes, num_populations = 1, state_size = 1, pool_size = 1, max_depth = 7, max_init_depth = 4, 
                 tournament_size = 7, init_method = "ramped", size_parsinomy = 1, migration_method = "ring", migration_period = 5, migration_percentage = 0.1, restart_iter_threshold = None):
        super().__init__(num_generations, population_size, fitness_function)
        self.expressions = expressions
        self.layer_sizes = layer_sizes
        self.pool = Pool(pool_size)
        self.pool.close()
        self.num_populations = num_populations
        self.state_size = state_size
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.init_method = init_method
        self.size_parsinomy = size_parsinomy

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
        self.selection_pressures = jnp.linspace(0.7,1.0,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(0.65,0.3,self.num_populations),jnp.linspace(0.3,0.5,self.num_populations),
                                         jnp.linspace(0.,0.2,self.num_populations),jnp.linspace(0.05,0.,self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(0.6,0.2,self.num_populations)

        self.mutation_probabilities = {}
        self.mutation_probabilities["mutate_operator"] = 0.5
        self.mutation_probabilities["delete_operator"] = 1.0
        self.mutation_probabilities["insert_operator"] = 0.5
        self.mutation_probabilities["mutate_constant"] = 1.0
        self.mutation_probabilities["mutate_leaf"] = 1.0
        self.mutation_probabilities["sample_subtree"] = 1.0
        self.mutation_probabilities["prepend_operator"] = 0.5
        self.mutation_probabilities["add_subtree"] = 1.0

        self.best_fitness_per_population = jnp.zeros((num_generations, self.num_populations))

    def initialize_population(self, key):
        #Sample a specified number of trees
        keys = jrandom.split(key, self.num_populations+1)
        self.pool.restart()
        populations = self.pool.amap(lambda x: initialization.sample_trees(keys[x], self.expressions, self.layer_sizes, self.population_size, self.max_init_depth, self.init_method), 
                                           range(self.num_populations))
        self.pool.close()
        self.pool.join()
        return populations.get()

    def evaluate_population(self, population, data):
        self.pool.restart()
        fitness = self.pool.amap(lambda x: self.evaluate(x, data) if x.fitness == None else x.fitness, helper_functions.flatten(population)) #Evaluate each solution parallely on a pool of workers
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

        self.best_fitness_per_population = self.best_fitness_per_population.at[self.current_generation].set(jnp.min(fitnesses, axis=1))
        best_solution_of_g = helper_functions.best_solution(population, fitnesses)
        best_fitness_of_g = self.evaluate(best_solution_of_g, data)

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

        self.current_generation += 1

        return fitnesses, population
    
    def evaluate(self, model, data):
        model_functions = model.tree_to_function(self.expressions)
        return self.fitness_function(model_functions, data)

    def sample_population(self, key, population):
        #Migrate individuals between populations every few generations
        if ((self.current_generation+1)%self.migration_period)==0:
            key, new_key = jrandom.split(key)
            population = migration.migrate_populations(population, self.migration_method, self.migration_size, new_key)

        self.last_restart = self.last_restart + 1
        restart = jnp.logical_and(self.last_restart>self.restart_iter_threshold, self.best_fitness_per_population[self.current_generation] >= 
                                  jnp.array([self.best_fitness_per_population[self.current_generation-self.restart_iter_threshold[i], i] for i in range(self.num_populations)]))
        self.last_restart = self.last_restart.at[restart].set(0)
            
        keys = jrandom.split(key, self.num_populations+1)
        self.pool.restart()
        #Evaluate each solution parallely on a pool of workers
        new_populations = self.pool.amap(lambda x: self.update_population(x, population[x], restart[x], keys[x]), range(self.num_populations))
        self.pool.close()
        self.pool.join()
        return new_populations.get()
    
    def update_population(self, index, population, restart, key):
        if restart:
            return initialization.sample_trees(key, self.expressions, self.layer_sizes, self.population_size, self.max_init_depth, self.init_method)
        else:
            return reproduction.next_population(population, key, self.expressions, self.layer_sizes, self.reproduction_type_probabilities[index], self.reproduction_probabilities[index], 
                                        self.mutation_probabilities, self.tournament_probabilities[index], self.tournament_size, self.max_depth, self.max_init_depth)
