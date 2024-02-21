import jax
import jax.numpy as jnp
import evosax
from algorithms.strategy_base import Strategy

class CMA_ES(Strategy):
    def __init__(self, num_generations, population_size, fitness_function, num_dims, key) -> None:
        super().__init__(num_generations, population_size, fitness_function)
        self.cma_strategy = evosax.CMA_ES(popsize = population_size, num_dims = num_dims)
        self.cma_state = self.cma_strategy.initialize(key)

    def initialize_population(self, key):
        population, self.cma_state = self.cma_strategy.ask(key, self.cma_state)
        return population

    def evaluate_population(self, population, data):
        fitnesses = jax.vmap(self.fitness_function, in_axes=[0,None])(population, data)
        self.cma_state = self.cma_strategy.tell(population, fitnesses, self.cma_state)

        best_idx = jnp.argmin(fitnesses)
        best_fitness_of_g = fitnesses[best_idx]
        best_solution_of_g = population[best_idx]

        if self.current_generation == 0:
            best_fitness = best_fitness_of_g
            best_solution = best_solution_of_g
        else:
            best_fitness = self.best_fitnesses[self.current_generation - 1]
            best_solution = self.best_solutions[self.current_generation - 1]

            if best_fitness_of_g < best_fitness:
                best_fitness = best_fitness_of_g
                best_solution = best_solution_of_g

        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)
        self.best_solutions.append(best_solution)

        self.current_generation += 1
        return fitnesses, population

    def sample_population(self, key, population):
        population, self.cma_state = self.cma_strategy.ask(key, self.cma_state)
        return population