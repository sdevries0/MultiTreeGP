import jax
from jax import Array
import jax.numpy as jnp
from jax.random import PRNGKey
import evosax

from typing import Tuple

from MultiTreeGP.algorithms.strategy_base import Strategy

class CMA_ES(Strategy):
    """Covariance matrix adaptation evolution strategy of symbolic expressions.

        Attributes:
            num_generations: The number of generations over which to evolve the population.
            population_size: Number of candidates in the population.
            fitness_function: Function that evaluates a candidate and assigns a fitness.
            current_generation: A reference to keep track of the current generation.
            best_solutions: Best solution at each generation.
            best_fitnesses: Best fitness at each generation.
            cma_strategy: Evosax implementation of CMA-ES.
            cma_state: State that controls the evolution of CMA-ES.

        """
    def __init__(self, num_generations, population_size, fitness_function, num_dims, key) -> None:
        super().__init__(num_generations, population_size, fitness_function)
        self.cma_strategy = evosax.CMA_ES(popsize = population_size, num_dims = num_dims, elite_ratio=0.1, sigma_init=3)
        self.cma_state = self.cma_strategy.initialize(key)

    def initialize_population(self, key: PRNGKey) -> list:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        population, self.cma_state = self.cma_strategy.ask(key, self.cma_state)
        return population

    def evaluate_population(self, population: Array, data: Tuple) -> Tuple[Array, list]:
        """Evaluates every candidate in population and assigns a fitness.

        :param population: Population of candidates
        :param data: The data required to evaluate the population.

        Returns: Fitness and evaluated population.
        """
        fitnesses = jax.vmap(self.fitness_function, in_axes=[0,None])(population, data)
        self.cma_state = self.cma_strategy.tell(population, fitnesses, self.cma_state)

        best_idx = jnp.argmin(fitnesses)
        best_fitness_of_g = fitnesses[best_idx]
        best_solution_of_g = population[best_idx]

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

        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)
        self.best_solutions.append(best_solution)

        return fitnesses, population

    def evolve_population(self, population: Array, key: PRNGKey) -> Array:
        """Creates a new population by evolving the current population.

        :param population: Population of candidates
        :param key: Random key.

        Returns: Population with new candidates.
        """  
        population, self.cma_state = self.cma_strategy.ask(key, self.cma_state)
        self.current_generation += 1
        return population