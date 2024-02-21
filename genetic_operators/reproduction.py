import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
import equinox as eqx

import genetic_operators.mutation as mutation
import genetic_operators.crossover as crossover
import miscellaneous.helper_functions as helper_functions
import genetic_operators.initialization as initialization
from miscellaneous.expression import Expression
from genetic_operators.simplification import trees_to_sympy
import genetic_operators.simplification as simplification

def tournament_selection(population: list, key: PRNGKey, tournament_probabilities, tournament_size):
    "Selects a candidate from a randomly selected tournament. Selection is based on fitness and the probability of being chosen given a rank"
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

def equal_trees(parent, child, layer_sizes, expressions):
    if parent == child:
        return True
    for i in range(len(layer_sizes)):
        for tree in range(layer_sizes[i]):
            if expressions[i].condition(child()[i][tree]):
                return True
    return False

def next_population(population: list, key: PRNGKey, expressions, layer_sizes, reproduction_type_probabilities, reproduction_probabilities, mutation_probabilities, 
                    tournament_probabilities, tournament_size, max_depth, max_init_depth):
    "Generates a new population by evolving the current population. After cross-over and mutation, the new trees are checked to be different from their parents."
    population_size = len(population)
    population.sort(key=lambda x: x.fitness)
    new_pop = []
    elite_size = int(population_size*0.1)
    for i in range(elite_size):
        new_pop.append(population[i])
    remaining_candidates = population_size - elite_size
    failed_mutations = 0

    while remaining_candidates>0: #Loop until new population has reached the desired size
        probs = reproduction_type_probabilities.copy()
        key, new_key1 = jrandom.split(key, 2)
        parent = tournament_selection(population, new_key1, tournament_probabilities, tournament_size)

        if remaining_candidates==1:
            probs = probs.at[0].set(0)

        key, new_key = jrandom.split(key)
        reproduction_type = jrandom.choice(new_key, jnp.arange(4), p=jnp.array(probs))
        if reproduction_type==0: #Cross-over
            key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
            partner = tournament_selection(population, new_key1, tournament_probabilities, tournament_size)

            cross_over_type = jrandom.choice(new_key2, jnp.arange(3), p=jnp.array([0.2,0.4,0.4])) #Sample a cross-over method    
            if cross_over_type == 0:
                offspring = crossover.tree_cross_over(parent, partner, reproduction_probabilities, layer_sizes, new_key3)
            elif cross_over_type == 1:
                offspring = crossover.uniform_cross_over(parent, partner, reproduction_probabilities, layer_sizes, new_key3)
            else:
                offspring = crossover.standard_cross_over(parent, partner, reproduction_probabilities, layer_sizes, new_key3)

            #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
            if helper_functions.tree_depth(offspring[0]) > max_depth or equal_trees(parent, offspring[0], layer_sizes, expressions):
                failed_mutations += 1
            else:
                #Append new trees to the new population
                offspring[0].reset_fitness()
                new_pop.append(offspring[0])
                remaining_candidates -= 1

            #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
            if helper_functions.tree_depth(offspring[1]) > max_depth or equal_trees(partner, offspring[1], layer_sizes, expressions):
                failed_mutations += 1
            else:
                #Append new trees to the new population
                offspring[1].reset_fitness()
                new_pop.append(offspring[1])
                remaining_candidates -= 1

        elif reproduction_type==1: #Mutation

            child = mutation.mutate_trees(parent, layer_sizes, new_key, reproduction_probabilities, mutation_probabilities, 
                                    expressions, max_init_depth)
            
            #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
            if (helper_functions.tree_depth(child) > max_depth) or equal_trees(parent, child, layer_sizes, expressions):
                failed_mutations += 1
            else:
                #Append new trees to the new population
                child.reset_fitness()
                new_pop.append(child)
                remaining_candidates -= 1

        elif reproduction_type==2: #Sample new trees
            key, new_key = jrandom.split(key)
            new_trees = initialization.sample_trees(new_key, expressions, layer_sizes, max_depth = max_init_depth, N = 1, 
                                                    init_method = "full")
            #Add new trees to the new population
            remaining_candidates -= 1
            new_pop.append(new_trees)

        elif reproduction_type==3: #simplification
            child = parent
            simplified = False
            for i in range(layer_sizes.shape[0]):
                for j in range(layer_sizes[i]):
                    simplified_tree = simplification.simplify_tree(parent()[i][j])
                    if simplified_tree != False:
                        child = eqx.tree_at(lambda t: t()[i][j], child, simplified_tree)
                        simplified = True
            if simplified:
                remaining_candidates -= 1
                new_pop.append(child)
            else:
                failed_mutations += 1
    return new_pop