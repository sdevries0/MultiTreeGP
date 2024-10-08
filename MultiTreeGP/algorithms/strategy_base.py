import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey

from typing import Callable, Tuple

class Strategy:
    """General base strategy class for evolutionary optimisation.

        Attributes:
            num_generations: The number of generations over which to evolve the population.
            population_size: Number of candidates in the population.
            fitness_function: Function that evaluates a candidate and assigns a fitness.
            current_generation: A reference to keep track of the current generation.
            best_solutions: Best solution at each generation.
            best_fitnesses: Best fitness at each generation.
        """
    def __init__(self, num_generations: int, population_size: int, fitness_function: Callable, candidate_shape: Tuple) -> None:
        self.num_generations = num_generations
        self.population_size = population_size
        self.fitness_function = fitness_function

        self.current_generation = 0
        self.best_fitnesses = jnp.zeros(num_generations)
        self.best_solutions = jnp.zeros((num_generations, *candidate_shape))

    def initialize_population(self, key: PRNGKey) -> list:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        pass

    def evaluate_population(self, population: list, data: Tuple) -> Tuple[Array, list]:
        """Evaluates every candidate in population and assigns a fitness.

        :param population: Population of candidates
        :param data: The data required to evaluate the population.

        Returns: Fitness and evaluated population.
        """
        pass

    def evolve_population(self, population: list, key: PRNGKey) -> list:
        """Creates a new population by evolving the current population.

        :param population: Population of candidates
        :param key: Random key.

        Returns: Population with new candidates.
        """
        pass

    def get_statistics(self, generation: int = None):
        """Returns best fitness and best solution.

        :param generation: Generation of which fitness and solution are required. If None, returns all fitnesses and solutions.

        Returns: Best fitness and best solution.
        """
        if generation is not None:
            return self.best_fitnesses[generation], self.best_solutions[generation]
        else:
            return self.best_fitnesses, self.best_solutions