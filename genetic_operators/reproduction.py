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

class Reproducer:
    """
    Class that manages the reproduction of a population of trees via genetic programming. At every generation reproduction rules are applied to the current population to produce offspring.
    
    Attributes:
        expressions: container with expressions that can be included in trees
        population_size (int): Determines the number of trees that can be placed in a subpopulation
        state_size (int): The number of hidden neurons in an individual
        num_populations (int): Number of independent subpopulations
        tournament_size (int): Size of a tournament used for selecting parents
        selection_pressures (Array[float]): Selection pressure for each subpopulation.
        tournament_probabilities (Array[float]): The probabilities for each of the ranked individuals to be selected. Depends on selection pressure of each subpopulation
        similarity_threshold (float): Threshold that determines when two trees are similar
        max_depth (int): The highest depth branches of trees are allowed to reach
        max_init_depth (int): The highest depth branches of trees are allowed to reach during initialization
        init_method (str): Method for initialization of trees
        unary_operators_prob (float): Probability of sampled unary operators
        reproduction_type_probabilities (Array[float]): The probabilities for each reproduction types. These are different for each subpopulation
        reproduction_probabilities (Array[float]): The probability of a tree to be mutated. This is different for each subpopulation
        mutation_probabilities (dict[str, float]): Probability for each type of mutation
    
    
    """
    def __init__(self, expressions: Expression, layer_sizes, population_size: int, num_populations: int, test_samples: float, max_init_depth: int = 5, 
                    max_depth: int = 10, tournament_size: int = 5, init_method: str = "ramped"):
        self.expressions = expressions
        self.layer_sizes = layer_sizes
        self.population_size = population_size
        self.num_populations = num_populations
        self.tournament_size = tournament_size
        self.test_samples = test_samples
        self.selection_pressures = jnp.linspace(0.7,1.0,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.init_method = init_method
        self.unary_operators_prob = 0.
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(0.65,0.3,self.num_populations),jnp.linspace(0.3,0.5,self.num_populations),
                                         jnp.linspace(0.0,0.2,self.num_populations),jnp.linspace(0.05,0.00,self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(0.6,0.2,self.num_populations)

        self.mutation_probabilities = {}
        self.mutation_probabilities["mutate_operator"] = 0.5
        self.mutation_probabilities["delete_operator"] = 0.5
        self.mutation_probabilities["insert_operator"] = 0.5
        self.mutation_probabilities["mutate_constant"] = 1.0
        self.mutation_probabilities["mutate_leaf"] = 1.0
        self.mutation_probabilities["sample_subtree"] = 1.0
        self.mutation_probabilities["prepend_operator"] = 0.5
        self.mutation_probabilities["add_subtree"] = 1.0

    def tournament_selection(self, population: list, population_index: int, key: PRNGKey):
        "Selects a candidate from a randomly selected tournament. Selection is based on fitness and the probability of being chosen given a rank"
        key1, key2 = jrandom.split(key)

        tournament = []
        #Sample solutions to include in the tournament
        tournament_indices = jrandom.choice(key1, self.population_size, shape=(self.tournament_size,), replace=False)
        for i in tournament_indices:
            tournament.append(population[i])
        #Sort on fitness
        tournament.sort(key=lambda x: x.fitness)
        #Sample tournament winner
        index = jrandom.choice(key2, self.tournament_size, p=self.tournament_probabilities[population_index])
        return tournament[index]
    
    def equal_trees(self, parent, child):
        if parent == child:
            return True
        for i in range(len(self.layer_sizes)):
            for tree in range(self.layer_sizes[i]):
                if self.expressions[i].condition(child()[i][tree]):
                    return True
        return False

    def next_population(self, population: list, population_index: int, key: PRNGKey):
        "Generates a new population by evolving the current population. After cross-over and mutation, the new trees are checked to be different from their parents."
        population.sort(key=lambda x: x.fitness)
        new_pop = []
        elite_size = int(self.population_size*0.1)
        for i in range(elite_size):
            new_pop.append(population[i])
        remaining_candidates = self.population_size - elite_size
        failed_mutations = 0

        while remaining_candidates>0: #Loop until new population has reached the desired size
            probs = self.reproduction_type_probabilities[population_index].copy()
            key, new_key1 = jrandom.split(key, 2)
            parent = self.tournament_selection(population, population_index, new_key1)

            if remaining_candidates==1:
                probs = probs.at[0].set(0)

            key, new_key = jrandom.split(key)
            reproduction_type = jrandom.choice(new_key, jnp.arange(4), p=jnp.array(probs))
            if reproduction_type==0: #Cross-over
                key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
                partner = self.tournament_selection(population, population_index, new_key1)
                cross_over_type = jrandom.choice(new_key2, jnp.arange(3), p=jnp.array([0.2,0.4,0.4])) #Sample a cross-over method    
                if cross_over_type == 0:
                    offspring = crossover.tree_cross_over(parent, partner, self.reproduction_probabilities[population_index], self.layer_sizes, new_key3)
                elif cross_over_type == 1:
                    offspring = crossover.uniform_cross_over(parent, partner, self.reproduction_probabilities[population_index], self.layer_sizes, new_key3)
                else:
                    offspring = crossover.standard_cross_over(parent, partner, self.reproduction_probabilities[population_index], self.layer_sizes, new_key3)

                #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
                if helper_functions.tree_depth(offspring[0]) > self.max_depth or self.equal_trees(parent, offspring[0]):
                    failed_mutations += 1
                else:
                    #Append new trees to the new population
                    offspring[0].reset_fitness()
                    new_pop.append(offspring[0])
                    remaining_candidates -= 1

                #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
                if helper_functions.tree_depth(offspring[1]) > self.max_depth or self.equal_trees(partner, offspring[1]):
                    failed_mutations += 1
                else:
                    #Append new trees to the new population
                    offspring[1].reset_fitness()
                    new_pop.append(offspring[1])
                    remaining_candidates -= 1

            elif reproduction_type==1: #Mutation

                child = mutation.mutate_trees(parent, self.layer_sizes, new_key, self.reproduction_probabilities[population_index], self.mutation_probabilities, 
                                      self.expressions, self.max_init_depth, unary_operators_prob = self.unary_operators_prob)
                
                #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
                if (helper_functions.tree_depth(child) > self.max_depth) or self.equal_trees(parent, child) :
                    failed_mutations += 1
                else:
                    #Append new trees to the new population
                    child.reset_fitness()
                    new_pop.append(child)
                    remaining_candidates -= 1

            elif reproduction_type==2: #Sample new trees
                key, new_key = jrandom.split(key)
                new_trees = initialization.sample_trees(new_key, self.expressions, self.layer_sizes, max_depth = self.max_init_depth, N = 1, 
                                                        init_method = "full", unary_operators_prob=self.unary_operators_prob)
                #Add new trees to the new population
                remaining_candidates -= 1
                new_pop.append(new_trees)

            elif reproduction_type==3: #simplification
                child = parent
                simplified = False
                for i in range(self.layer_sizes.shape[0]):
                    for j in range(self.layer_sizes[i]):
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

    def sample_trees(self, key):
        #Sample a specified number of trees
        return initialization.sample_trees(key, self.expressions, self.layer_sizes, self.population_size, self.max_init_depth, self.init_method, unary_operators_prob = self.unary_operators_prob)