import jax.numpy as jnp

class Strategy:
    def __init__(self, num_generations, population_size, fitness_function) -> None:
        self.num_generations = num_generations
        self.population_size = population_size
        self.fitness_function = fitness_function

        self.current_generation = 0
        self.best_solutions = []
        self.best_fitnesses = jnp.zeros(num_generations)

    def initialize_population(self, key):
        pass

    def evaluate_population(self, population, data):
        pass

    def sample_population(self, population, key):
        pass

    def get_statistics(self, generation = None):
        if generation is not None:
            return self.best_fitnesses[generation], self.best_solutions[generation]
        else:
            return self.best_fitnesses, self.best_solutions