import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
import equinox as eqx

import MultiTreeGP.genetic_operators.mutation as mutation
import MultiTreeGP.genetic_operators.crossover as crossover
import MultiTreeGP.genetic_operators.initialization as initialization
import MultiTreeGP.genetic_operators.simplification as simplification
from MultiTreeGP.networks.tree_policy import TreePolicy


def tree_depth(tree: TreePolicy) -> int:
    """Returns highest depth of tree.

    :param tree: Tree policy.
    :returns: Highest depth of the nodes in the tree.
    """
    flat_tree = jax.tree_util.tree_leaves_with_path(tree)
    return jnp.max(jnp.array([len(node[0]) for node in flat_tree]))

def tournament_selection(population: list, key: PRNGKey, tournament_probabilities: Array, tournament_size: int) -> TreePolicy:
    """Selects a candidate from a randomly selected tournament. Selection is based on fitness and the probability of being chosen given a rank.

    :param population: Population of tree policies.
    :param key: Random key.
    :param tournament_probabilities: Probability to be selected of each rank in the tournament.
    :param tournament_size: Size of the tournament.
    :returns: Tree policy that wins the tournament.
    """
    key1, key2 = jrandom.split(key)

    tournament = []
    #Sample solutions to include in the tournament
    tournament_indices = jrandom.choice(key1, len(population), shape=(tournament_size,), replace=False)
    for i in tournament_indices:
        tournament.append(population[i])
    #Sort on fitness
    tournament.sort(key=lambda x: x.fitness)
    #Sample tournament winner
    index = jrandom.choice(key2, tournament_size, p=tournament_probabilities)
    return tournament[index]

def invalid_trees(parent: TreePolicy, child: TreePolicy, layer_sizes: Array, expressions: list) -> bool:
    """If the child is equal to its parent or violates a condition, do not accept it in the new population.

    :param parent: Parent tree policy.
    :param child: Child tree policy
    :param layer_sizes: Size of each layer in a tree policy.
    :param expressions: Expressions for each layer in a tree policy.
    :returns: Boolean indicating whether the child should be accepted in the new population.
    """
    # if parent == child:
    #     return True
    equals = True

    for i in range(len(layer_sizes)):
        for tree in range(layer_sizes[i]):
            if not expressions[i].condition(child()[i][tree]):
                return True
            if parent()[i][tree] != child()[i][tree]:
                equals = False
    return equals

def next_population(population: list, 
                    key: PRNGKey, 
                    expressions: list, 
                    layer_sizes: Array, 
                    reproduction_type_probabilities: Array, 
                    reproduction_probability: float, 
                    mutation_probabilities: dict, 
                    tournament_probabilities: Array, 
                    tournament_size: int, 
                    max_depth: int, 
                    max_init_depth: int, 
                    elite_percentage: float,
                    leaf_sd: float
                ) -> list:
    """Generates a new population by evolving the current population. After cross-over and mutation, the new trees are checked to be different from their parents.

    :param population: Population of tree policies.
    :param key: Random key.
    :param expressions: Expressions for each layer in a tree policy.
    :param layer_sizes: Size of each layer in a tree policy.
    :param reproduction_type_probabilities: Probabilities for each of reproduction types. 
    :param reproduction_probability: Probability of a tree to be adapted in a tree policy.
    :param mutation_probabilities: Probabilities of the mutation functions.
    :param tournament_probabilities: Probability to be selected of each rank in the tournament.
    :param tournament_size: Size of the tournament.
    :param max_depth: Highest depth of a tree.
    :param max_init_depth: Highest depth of a tree at initialization.
    :param elite_percentage: Percentage of population that is the elite.
    :param leaf_sd: Standard deviation for sampling constants.
    :returns: New population of tree policies.
    """
    population_size = len(population)
    population.sort(key=lambda x: x.fitness)
    new_pop = []
    #Keep elite candidates in population
    elite_size = int(population_size*elite_percentage)
    for i in range(elite_size):
        new_pop.append(population[i])
    remaining_candidates = population_size - elite_size
    failed_mutations = 0

    while remaining_candidates>0: #Loop until new population has reached the desired size
        probs = reproduction_type_probabilities.copy()
        key, new_key1 = jrandom.split(key, 2)
        #Select parent
        parent = tournament_selection(population, new_key1, tournament_probabilities, tournament_size)

        if remaining_candidates==1:
            probs = probs.at[0].set(0)

        key, new_key = jrandom.split(key)
        reproduction_type = jrandom.choice(new_key, jnp.arange(4), p=jnp.array(probs))

        if reproduction_type==0: #Cross-over
            key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
            #Select second parent
            partner = tournament_selection(population, new_key1, tournament_probabilities, tournament_size)

            #Sample a cross-over method
            cross_over_type = jrandom.choice(new_key2, jnp.arange(3), p=jnp.array([0.0 if jnp.sum(layer_sizes) > 1 else 0,0.0,0.4]))    
            if cross_over_type == 0:
                offspring = crossover.tree_cross_over(parent, partner, reproduction_probability, layer_sizes, new_key3)
            elif cross_over_type == 1:
                offspring = crossover.uniform_cross_over(parent, partner, reproduction_probability, layer_sizes, new_key3)
            else:
                offspring = crossover.standard_cross_over(parent, partner, reproduction_probability, layer_sizes, new_key3, expressions)

            #If a tree policy remain the same or one of the trees exceeds the max depth, cross-over has failed
            if tree_depth(offspring[0]) > max_depth or invalid_trees(parent, offspring[0], layer_sizes, expressions):
                failed_mutations += 1
            else:
                #Append new tree policy to the new population
                key, new_key1, new_key2 = jrandom.split(key, 3)
                child1 = offspring[0]
                child1.reset_fitness()
                if jrandom.uniform(new_key1)<0.1:
                    for i in range(layer_sizes.shape[0]):
                        for j in range(layer_sizes[i]):
                            simplified_tree = simplification.simplify_tree(child1()[i][j])
                            if simplified_tree:
                                child1 = eqx.tree_at(lambda t: t()[i][j], child1, simplified_tree)
                new_pop.append(child1)
                remaining_candidates -= 1

                # print(f"Crossover type {cross_over_type}, parent: {parent}, child: {offspring[0]}")

            #If a tree policy remain the same or one of the trees exceeds the max depth, cross-over has failed
            if tree_depth(offspring[1]) > max_depth or invalid_trees(partner, offspring[1], layer_sizes, expressions):
                failed_mutations += 1
            else:
                #Append new tree policy to the new population
                child2 = offspring[1]
                child2.reset_fitness()
                if jrandom.uniform(new_key2)<0.1:
                    for i in range(layer_sizes.shape[0]):
                        for j in range(layer_sizes[i]):
                            simplified_tree = simplification.simplify_tree(child2()[i][j])
                            if simplified_tree:
                                child2 = eqx.tree_at(lambda t: t()[i][j], child2, simplified_tree)
                new_pop.append(child2)
                remaining_candidates -= 1

                # print(f"Crossover type {cross_over_type}, parent: {partner}, child: {offspring[1]}")

        elif reproduction_type==1: #Mutation
            key, new_key1, new_key2 = jrandom.split(key, 3)
            child = mutation.mutate_trees(parent, layer_sizes, new_key1, reproduction_probability, mutation_probabilities, 
                                    expressions, max_init_depth, leaf_sd)
            
            #If a tree policy remain the same or one of the trees exceeds the max depth, mutation has failed
            if (tree_depth(child) > max_depth) or invalid_trees(parent, child, layer_sizes, expressions):
                failed_mutations += 1
            else:
                #Append new tree policy to the new population
                child.reset_fitness()
                if jrandom.uniform(new_key2)<0.1:
                    for i in range(layer_sizes.shape[0]):
                        for j in range(layer_sizes[i]):
                            simplified_tree = simplification.simplify_tree(child()[i][j])
                            if simplified_tree:
                                child = eqx.tree_at(lambda t: t()[i][j], child, simplified_tree)
                new_pop.append(child)
                remaining_candidates -= 1
                # print(f"Mutation, parent: {parent}, child: {child}")

        elif reproduction_type==2: #Sample new tree policy
            key, new_key = jrandom.split(key)
            new_trees = initialization.sample_trees(new_key, expressions, layer_sizes, max_depth = max_init_depth, N = 1, 
                                                    init_method = "ramped", leaf_sd = leaf_sd)
            #Add new tree policy to the new population
            remaining_candidates -= 1
            new_pop.append(new_trees)

        elif reproduction_type==3: #Simplification of a tree policy
            print("simpli")
            child = parent
            simplified = False
            for i in range(layer_sizes.shape[0]):
                for j in range(layer_sizes[i]):
                    simplified_tree = simplification.simplify_tree(parent()[i][j])
                    if simplified_tree:
                        child = eqx.tree_at(lambda t: t()[i][j], child, simplified_tree)
                        simplified = True
            #If at least one tree is simplified, add simplified tree policy to new population
            if simplified:
                remaining_candidates -= 1
                new_pop.append(child)
            else:
                failed_mutations += 1
    return new_pop