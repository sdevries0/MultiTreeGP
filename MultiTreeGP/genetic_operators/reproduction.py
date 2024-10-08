import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey

def evolve_trees(parent1, parent2, keys, type, reproduction_probability, reproduction_functions):
    child1, child2, mutate_functions = jax.lax.switch(type, reproduction_functions, parent1, parent2, keys, reproduction_probability)

    return child1, child2, mutate_functions

def tournament_selection(population, fitness, key, tournament_probabilities, tournament_size, population_indices):
    tournament_key, winner_key = jr.split(key)
    indices = jr.choice(tournament_key, population_indices, shape=(tournament_size,))
    index = jr.choice(winner_key, indices[jnp.argsort(fitness[indices])], p=tournament_probabilities)
    return population[index]

def evolve_population(population, fitness, key, reproduction_type_probabilities, reproduction_probability, tournament_probabilities, population_indices, population_size, tournament_size, num_trees, elite_size, reproduction_functions):
    left_key, right_key, repro_key, cx_key = jr.split(key, 4)
    elite = population[jnp.argsort(fitness)[:elite_size]]
    left_parents = jax.vmap(tournament_selection, in_axes=[None, None, 0, None, None, None])(population, fitness, jr.split(left_key, (population_size - elite_size)//2), tournament_probabilities, tournament_size, population_indices)
    right_parents = jax.vmap(tournament_selection, in_axes=[None, None, 0, None, None, None])(population, fitness, jr.split(right_key, (population_size - elite_size)//2), tournament_probabilities, tournament_size, population_indices)
    reproduction_type = jr.choice(repro_key, jnp.arange(3), shape=((population_size - elite_size)//2,), p=reproduction_type_probabilities)
    left_children, right_children, mutate_functions = jax.vmap(evolve_trees, in_axes=[0, 0, 0, 0, None, None])(left_parents, right_parents, jr.split(cx_key, ((population_size - elite_size)//2, num_trees, 2)), reproduction_type, reproduction_probability, reproduction_functions)
    evolved_population = jnp.concatenate([elite, left_children, right_children], axis=0)
    return evolved_population, left_parents, right_parents, left_children, right_children, reproduction_type, mutate_functions

def migrate_population(receiver, sender, receiver_fitness, sender_fitness, migration_size, population_indices):
    sorted_receiver = receiver[jnp.argsort(receiver_fitness, descending=True)]
    sorted_sender = sender[jnp.argsort(sender_fitness, descending=False)]
    return jnp.where((population_indices < migration_size)[:,None,None,None], sorted_sender, sorted_receiver)

def evolve_populations(jit_evolve_population, populations, fitness, key, current_generation, migration_period, migration_size, reproduction_type_probabilities, reproduction_probabilities, tournament_probabilities):
    num_populations, population_size, num_trees, _, _ = populations.shape
    population_indices = jnp.arange(population_size)
    populations = jax.lax.select((num_populations > 1) & (((current_generation+1)%migration_period) == 0), 
                                    jax.vmap(migrate_population, in_axes=[0, 0, 0, 0, None, None])(populations, jnp.roll(populations, 1, axis=0), fitness, jnp.roll(fitness, 1, axis=0), migration_size, population_indices), 
                                    populations)
    new_population, left_parents, right_parents, left_children, right_children, reproduction_type, mutate_functions = jax.vmap(jit_evolve_population, in_axes=[0, 0, 0, 0, 0, 0, None])(populations, fitness, jr.split(key, num_populations), reproduction_type_probabilities, 
                        reproduction_probabilities, tournament_probabilities, population_indices)
    return new_population, left_parents, right_parents, left_children, right_children, reproduction_type, mutate_functions