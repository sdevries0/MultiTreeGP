import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
import equinox as eqx

import mutation, crossover, miscellaneous, initialization
from expression import Expression

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
    def __init__(self, expressions: Expression, population_size: int, state_size: int, num_populations: int, max_init_depth: int = 5, max_depth: int = 10, tournament_size: int = 5, similarity_threshold: float = 0.2, init_method: str = "ramped"):
        self.expressions = expressions
        self.population_size = population_size
        self.state_size = state_size
        self.num_populations = num_populations
        self.tournament_size = tournament_size
        self.selection_pressures = jnp.linspace(0.7,1.0,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        self.similarity_threshold = similarity_threshold
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.init_method = init_method
        self.unary_operators_prob = 0.0
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(0.7,0.4,self.num_populations),jnp.linspace(0.3,0.4,self.num_populations),
                                         jnp.linspace(0.0,0.2,self.num_populations),jnp.linspace(0.0,0.0,self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(0.8,0.2,self.num_populations)

        self.mutation_probabilities = {}
        self.mutation_probabilities["mutate_operator"] = 1.0
        self.mutation_probabilities["delete_operator"] = 0.5
        self.mutation_probabilities["insert_operator"] = 1.0
        self.mutation_probabilities["mutate_constant"] = 1.0
        self.mutation_probabilities["mutate_leaf"] = 1.0
        self.mutation_probabilities["sample_subtree"] = 1.0
        self.mutation_probabilities["prepend_operator"] = 0.5
        self.mutation_probabilities["add_subtree"] = 1.0
        self.mutation_probabilities["simplify_tree"] = 0.5

    def similarity(self, tree_a: list, tree_b: list):
        #Computes the similarity of two trees. 
        if len(tree_a)==1 and len(tree_b)==1:
            if isinstance(tree_a[0], jax.numpy.ndarray) and isinstance(tree_b[0], jax.numpy.ndarray):
                return 1
            if tree_a==tree_b:
                return 1
            return 0
        if len(tree_a) != len(tree_b):
            return 0
        if len(tree_a) == 2:
            return int(tree_a[0] == tree_b[0]) + self.similarity(tree_a[1], tree_b[1])
        #Compute the similarity for both permutations of the children nodes
        return int(tree_a[0] == tree_b[0]) + max(self.similarity(tree_a[1], tree_b[1]) + self.similarity(tree_a[2], tree_b[2]), self.similarity(tree_a[1], tree_b[2]) + self.similarity(tree_a[2], tree_b[1]))

    def similarities(self, tree_a: list, tree_b: list):
        #Accumulates the similarities for each pair of trees in two candidates
        similarity = 0
        for i in range(self.state_size):
            similarity += self.similarity(tree_a()[i],tree_b()[i])/min(len(jax.tree_util.tree_leaves(tree_a()[i])),len(jax.tree_util.tree_leaves(tree_b()[i])))
        similarity += self.similarity(tree_a.readout_tree,tree_b.readout_tree)/min(len(jax.tree_util.tree_leaves(tree_a.readout_tree)),len(jax.tree_util.tree_leaves(tree_b.readout_tree)))
        return similarity/(self.state_size+1)

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

    def next_population(self, population: list, mean_fitness: float, population_index: int, key: PRNGKey):
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
            key, new_key1, new_key2 = jrandom.split(key, 3)
            trees_a = self.tournament_selection(population, population_index, new_key1)
            trees_b = self.tournament_selection(population, population_index, new_key2)

            similarity = self.similarities(trees_a, trees_b)
            if trees_a.fitness > mean_fitness and trees_b.fitness > mean_fitness:
                probs = [0,0.5,0.5,0] #Do not apply cross-over if both trees have poor fitness

            elif similarity > self.similarity_threshold:
                probs = [0,0.6,0.3,0.1] #Do not apply crossover if trees are similar

            elif remaining_candidates==1:
                probs = probs.at[0].set(0)

            key, new_key = jrandom.split(key)
            reproduction_type = jrandom.choice(new_key, jnp.arange(4), p=jnp.array(probs))
                
            if reproduction_type==0: #Cross-over
                key, new_key1, new_key2 = jrandom.split(key, 3)
                cross_over_type = jrandom.choice(new_key1, jnp.arange(3)) #Sample a cross-over method           
                if cross_over_type == 0:
                    new_trees_a, new_trees_b = crossover.tree_cross_over(trees_a, trees_b, self.reproduction_probabilities[population_index], self.state_size, new_key2)
                elif cross_over_type == 1:
                    new_trees_a, new_trees_b = crossover.uniform_cross_over(trees_a, trees_b, self.reproduction_probabilities[population_index], self.state_size, new_key2)
                else:
                    new_trees_a, new_trees_b = crossover.standard_cross_over(trees_a, trees_b, self.reproduction_probabilities[population_index], self.state_size, new_key2)

                #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
                if eqx.tree_equal(trees_a, new_trees_a) or new_trees_a in population or (miscellaneous.tree_depth(new_trees_a) > self.max_depth):
                    failed_mutations += 1
                else:
                    #Append new trees to the new population
                    new_pop.append(new_trees_a)
                    remaining_candidates -= 1

                #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
                if eqx.tree_equal(trees_b, new_trees_b) or new_trees_b in population or (miscellaneous.tree_depth(new_trees_b) > self.max_depth):
                    failed_mutations += 1
                else:
                    #Append new trees to the new population
                    new_pop.append(new_trees_b)
                    remaining_candidates -= 1

            elif reproduction_type==1: #Mutation
                key, new_key = jrandom.split(key)
                mutate_bool = jrandom.bernoulli(new_key, p = self.reproduction_probabilities[population_index], shape=(self.state_size+1,))
                while jnp.sum(mutate_bool)==0: #Make sure that at least one tree is mutated
                    key, new_key = jrandom.split(key)
                    mutate_bool = jrandom.bernoulli(new_key, p = self.reproduction_probabilities[population_index], shape=(self.state_size+1,))

                new_trees_a = trees_a
                for i in range(self.state_size):
                    if mutate_bool[i]:
                        key, new_key = jrandom.split(key)
                        new_tree = mutation.mutate_tree(self.mutation_probabilities, self.expressions, trees_a()[i], new_key, self.max_init_depth)
                        new_trees_a = eqx.tree_at(lambda t: t()[i], new_trees_a, new_tree)
                if mutate_bool[-1]: #Mutate readout tree
                    key, new_key = jrandom.split(key)
                    new_trees_a = eqx.tree_at(lambda t: t.readout_tree, new_trees_a, mutation.mutate_tree(self.mutation_probabilities, self.expressions, new_trees_a.readout_tree, new_key, self.max_init_depth, readout=True))
                
                #If a tree remain the same or one of the trees exceeds the max depth or has already been added to the new population, cross-over has failed
                if eqx.tree_equal(trees_a, new_trees_a) or new_trees_a in population or (miscellaneous.tree_depth(new_trees_a) > self.max_depth):
                    failed_mutations += 1
                else:
                    #Append new trees to the new population
                    new_pop.append(new_trees_a)
                    remaining_candidates -= 1

            elif reproduction_type==2: #Sample new trees
                key, new_key = jrandom.split(key)
                new_trees = initialization.sample_trees(new_key, self.expressions,self.state_size, max_depth=self.max_init_depth, N=1, init_method="full")
                #Add new trees to the new population
                remaining_candidates -= 1
                new_pop.append(new_trees)
            elif reproduction_type==3: #Reproduction
                remaining_candidates -= 1
                #Add new trees to the new population
                new_pop.append(trees_a)
        
        return new_pop

    def sample_trees(self, key, population_size, num_populations):
        #Sample a specified number of trees
        return initialization.sample_trees(key, self.expressions, self.state_size, self.max_init_depth, population_size, num_populations, self.init_method)