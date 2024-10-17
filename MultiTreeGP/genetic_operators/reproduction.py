import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
from typing import Callable

def evolve_trees(parent1: Array, 
                 parent2: Array, 
                 keys: Array, 
                 type: int, 
                 reproduction_probability: float, 
                 reproduction_functions: list[Callable]) -> Array:
    """
    Applies reproduction function to pair of candidates
    
    :param parent1
    :param parent2
    :param keys
    :param reproduction_probability: Probability of a tree to be mutated
    :param reproduction_functions: Functions that be applied for reproduction

    Returns: Pair of reproduced candidates
    """
    child1, child2 = jax.lax.switch(type, reproduction_functions, parent1, parent2, keys, reproduction_probability)

    return child1, child2

def tournament_selection(population: Array, 
                         fitness: Array, 
                         key: PRNGKey, 
                         tournament_probabilities: Array, 
                         tournament_size: int, 
                         population_indices: Array) -> Array:
    """
    Selects a candidate for reproduction from a tournament

    :param population: Population of candidates
    :param fitness: Fitness of candidates
    :param tournament_probabilities: Probability of each of the ranks in the tournament to be selected for reproduction
    :param tournament_size
    :param population_indices

    Returns: Candidate that won the tournament
    """
    tournament_key, winner_key = jr.split(key)
    indices = jr.choice(tournament_key, population_indices, shape=(tournament_size,))
    index = jr.choice(winner_key, indices[jnp.argsort(fitness[indices])], p=tournament_probabilities)
    return population[index]

def evolve_population(population: Array, 
                      fitness: Array, 
                      key: PRNGKey, 
                      reproduction_type_probabilities: Array, 
                      reproduction_probability: float, 
                      tournament_probabilities: Array, 
                      population_indices: Array, 
                      population_size: int, 
                      tournament_size: int, 
                      num_trees: int, 
                      elite_size: int, 
                      reproduction_functions: list[Callable]) -> Array:
    """
    Reproduces pairs of candidates to obtain a new population

    :param population: Population of candidates
    :param fitness: Fitness of candidates
    :param key
    :param reproduction_type_probability: Probability of each reproduction function to be applied
    :param reproduction_probability: Probability of a tree to be mutated
    :param tournament_probabilities: Probability of each of the ranks in the tournament to be selected for reproduction
    :param population indices
    :param population_size
    :param tournament_size
    :param num_trees: Number of trees in a candidate
    :param elite_size: Number of candidates that remain in the new population without reproduction
    :param reproduction_functions: Functions that be applied for reproduction
    
    Returns: Evolved population
    """
    left_key, right_key, repro_key, cx_key = jr.split(key, 4)
    elite = population[jnp.argsort(fitness)[:elite_size]]

    left_parents = jax.vmap(tournament_selection, in_axes=[None, None, 0, None, None, None])(population, 
                                                                                             fitness, 
                                                                                             jr.split(left_key, (population_size - elite_size)//2), 
                                                                                             tournament_probabilities, 
                                                                                             tournament_size, 
                                                                                             population_indices)
    
    right_parents = jax.vmap(tournament_selection, in_axes=[None, None, 0, None, None, None])(population, 
                                                                                              fitness, 
                                                                                              jr.split(right_key, (population_size - elite_size)//2), 
                                                                                              tournament_probabilities, 
                                                                                              tournament_size, 
                                                                                              population_indices)
    
    reproduction_type = jr.choice(repro_key, jnp.arange(3), shape=((population_size - elite_size)//2,), p=reproduction_type_probabilities)

    left_children, right_children = jax.vmap(evolve_trees, in_axes=[0, 0, 0, 0, None, None])(left_parents, 
                                                                                             right_parents, 
                                                                                             jr.split(cx_key, ((population_size - elite_size)//2, num_trees, 2)), 
                                                                                             reproduction_type, 
                                                                                             reproduction_probability, 
                                                                                             reproduction_functions)
    
    evolved_population = jnp.concatenate([elite, left_children, right_children], axis=0)
    return evolved_population

def migrate_population(receiver: Array, 
                       sender: Array, 
                       receiver_fitness: Array, 
                       sender_fitness: Array, 
                       migration_size: int, 
                       population_indices: Array) -> Array:
    """
    Unfit candidates from one population are replaced with fit candidates from another population

    :param receiver: Population that receives new candidates
    :param sender: Population that sends fit candidates
    :param receiver_fitness: Fitness of the candidates in the receiving population
    :param sender_fitness: Fitness of the candidates in the sending population
    :param migration_size: How many candidates are migrated to new population
    :param population_indices

    Returns: Population after migration
    
    """
    sorted_receiver = receiver[jnp.argsort(receiver_fitness, descending=True)]
    sorted_sender = sender[jnp.argsort(sender_fitness, descending=False)]
    return jnp.where((population_indices < migration_size)[:,None,None,None], sorted_sender, sorted_receiver)

def evolve_populations(jit_evolve_population: Callable, 
                       populations: Array, 
                       fitness: Array, 
                       key: PRNGKey, current_generation: int, 
                       migration_period: int, 
                       migration_size: int, 
                       reproduction_type_probabilities: Array, 
                       reproduction_probabilities: Array, 
                       tournament_probabilities: Array) -> Array:
    """
    Evolves each population independently

    :param jit_evolve_population: Function for evolving trees that is jittable and parallelizable
    :param population: Populations of candidates
    :param fitness: Fitness of candidates
    :param key
    :param migration_period: After how many generations migration occurs
    :param migration_size: How many candidates are migrated to new population
    :param reproduction_type_probabilities: Probability of each reproduction function to be applied
    :param reproduction_probabilities: Probability of a tree to be mutated
    :param tournament_probabilities: Probability of each of the ranks in the tournament to be selected for reproduction

    Returns: Evolved populations
    """
    num_populations, population_size, num_trees, _, _ = populations.shape
    population_indices = jnp.arange(population_size)

    populations = jax.lax.select((num_populations > 1) & (((current_generation+1)%migration_period) == 0), 
                                    jax.vmap(migrate_population, in_axes=[0, 0, 0, 0, None, None])(populations, 
                                                                                                   jnp.roll(populations, 1, axis=0), 
                                                                                                   fitness, 
                                                                                                   jnp.roll(fitness, 1, axis=0), 
                                                                                                   migration_size, 
                                                                                                   population_indices), 
                                    populations)
    
    new_population = jax.vmap(jit_evolve_population, in_axes=[0, 0, 0, 0, 0, 0, None])(populations, 
                                                                                       fitness, 
                                                                                       jr.split(key, num_populations), 
                                                                                       reproduction_type_probabilities, 
                                                                                       reproduction_probabilities, 
                                                                                       tournament_probabilities, 
                                                                                       population_indices)
    return new_population