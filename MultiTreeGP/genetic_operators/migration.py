import jax
import jax.numpy as jnp
import jax.random as jrandom
import copy
from jax.random import PRNGKey

def migrate_trees(sender: list, receiver: list, migration_size: int, key: PRNGKey) -> list:
    """Selects individuals that will replace randomly selected indivuals in a receiver distribution.

    :param sender: Population from which individuals are selected. 
    :param receiver: Population in which indivuals are added.
    :param migration_size: Number of candidates to migrate.
    :param key: Random key.
    :returns: New population.
    """
    population_size = len(sender)
    sender.sort(key=lambda x: x.fitness)
    sender = sender[:migration_size]

    receiver_distribution = jnp.array([p.fitness for p in receiver]) #Select unfit individuals to replace with higher probability
    receiver_indices = jrandom.choice(key, population_size, shape=(migration_size,), p=receiver_distribution, replace=False)

    new_population = receiver

    #Insert migrated individuals
    for i in range(migration_size):
        new_population[receiver_indices[i]] = sender[i]

    return new_population

def migrate_populations(populations: list, migration_method: str, migration_size: int, key: PRNGKey) -> list:
    """Manages the migration between pairs of populations.

    :param populations: Populations of tree policies. 
    :param migration_method: Method for deciding the migration directions.
    :param migration_size: Number of candidates to migrate.
    :param key: Random key.
    :returns: Mutated tree.
    """
    assert (migration_method=="ring") or (migration_method=="random"), "This method is not implemented"

    num_populations = len(populations)
    if num_populations==1: #No migration possible
        return populations

    populations_copy = copy.copy(populations)

    if migration_method=="ring":
        for pop in range(num_populations):
            #Determine destination and sender
            destination = populations_copy[pop]
            sender = populations_copy[(pop+1)%num_populations]
            key, new_key = jrandom.split(key)
            populations[pop] = migrate_trees(sender, destination, migration_size, new_key)
    
    elif migration_method=="random":
        key, new_key = jrandom.split(key)
        permutation = jrandom.permutation(new_key, jnp.arange(num_populations))
        for pop in range(num_populations):
            #Determine destination and sample sender
            destination = populations_copy[pop]
            sender = populations_copy[permutation[pop]]
            key, new_key = jrandom.split(key)
            populations[pop] = migrate_trees(sender, destination, migration_size, new_key)

    return populations