import jax
import jax.numpy as jnp
import jax.random as jrandom
from expression import Expression
import networkTrees

def sample_leaf(key, expressions: Expression, sd: float = 1.0):
    "Samples a random leaf. The leaf is either a contant or a variable"
    key, new_key = jrandom.split(key)
    leaf_type = jrandom.uniform(new_key, shape=())
    if leaf_type<0.4:
        key, new_key = jrandom.split(key)
        return [sd*jrandom.normal(new_key)] #constant
    elif leaf_type<0.6: 
        key, new_key = jrandom.split(key)
        return [expressions.variables[jrandom.randint(new_key, shape=(), minval=0, maxval=len(expressions.variables))]] #observed variable
    elif leaf_type<0.8:
        key, new_key = jrandom.split(key)
        return [expressions.state_variables[jrandom.randint(new_key, shape=(), minval=0, maxval=len(expressions.state_variables))]] #hidden state variable
    elif leaf_type<0.9:
        key, new_key = jrandom.split(key)
        return [expressions.control_variables[jrandom.randint(new_key, shape=(), minval=0, maxval=len(expressions.control_variables))]] #control variable
    else:
        return ['target'] #target

def grow_node(key, expressions: Expression, depth: int, unary_operators_prob: float = 0.0):
    "Generates a random node that can contain leaves at lower depths"
    if depth == 1: #If depth is reached, a leaf is sampled
        key, new_key = jrandom.split(key)
        return sample_leaf(new_key, expressions, sd=3)
    
    key, new_key = jrandom.split(key)
    leaf_type = jrandom.choice(new_key, 3, p=jnp.array([0.3,unary_operators_prob,0.5])) #Sample the type of the leaf
    tree = []
    if leaf_type == 0: #leaf
        key, new_key = jrandom.split(key)
        return sample_leaf(new_key, expressions, sd=3)
    elif leaf_type == 1: #unary operator
        key, new_key1, new_key2 = jrandom.split(key, 3)
        tree.append(expressions.unary_operators[jrandom.randint(new_key1, shape=(), minval=0, maxval=len(expressions.unary_operators))])
        tree.append(grow_node(new_key2, expressions, depth-1))
        return tree
    elif leaf_type == 2: #binary operator
        key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
        tree.append(expressions.binary_operators[jrandom.randint(new_key1, shape=(), minval=0, maxval=len(expressions.binary_operators))])
        tree.append(grow_node(new_key2, expressions, depth-1))
        tree.append(grow_node(new_key3, expressions, depth-1))
        return tree
    
def full_node(key, expressions: Expression, depth: int, unary_operators_prob: float = 0.0):
    "Generates a random node that can only have leaves at the highest depth"
    if depth == 1: #If depth is reached, a leaf is sampled
        key, new_key = jrandom.split(key)
        return sample_leaf(new_key, expressions, sd=3)
    
    key, new_key = jrandom.split(key)
    leaf_type = jrandom.uniform(new_key, shape=()) #Sample the type of the leaf
    tree = []
    if leaf_type > 1-unary_operators_prob: #unary operator
        key, new_key1, new_key2 = jrandom.split(key, 3)
        tree.append(expressions.unary_operators[jrandom.randint(new_key1, shape=(), minval=0, maxval=len(expressions.unary_operators))])
        tree.append(full_node(new_key2, expressions, depth-1))
        return tree
    else: #binary operator
        key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
        tree.append(expressions.binary_operators[jrandom.randint(new_key1, shape=(), minval=0, maxval=len(expressions.binary_operators))])
        tree.append(full_node(new_key2, expressions, depth-1))
        tree.append(full_node(new_key3, expressions, depth-1))
        return tree

def sample_readout_leaf(key, expressions: Expression, sd: float = 1.0):
    "Samples a random leaf for the readout tree. The leaf is either a variable or a constant. Observations and control variables are excluded"
    key, new_key = jrandom.split(key)
    leaf_type = jrandom.uniform(new_key, shape=())
    if leaf_type<0.4:
        key, new_key = jrandom.split(key)
        return [sd*jrandom.normal(new_key)] #constant
    elif leaf_type<0.9:
        key, new_key = jrandom.split(key)
        return [expressions.state_variables[jrandom.randint(new_key, shape=(), minval=0, maxval=len(expressions.state_variables))]] #Hidden state variable
    else:
        return ['target'] #Target

def sample_readout_tree(key, expressions: Expression, depth: int, unary_operators_prob: float = 0.0):
    "Generates a random node for the readout tree that can contain leaves at lower depths"
    if depth == 1:
        key, new_key = jrandom.split(key)
        return sample_readout_leaf(new_key, expressions, sd=3) #If depth is reached, a leaf is sampled
    key, new_key = jrandom.split(key)
    leaf_type = jrandom.choice(new_key, 3, p=jnp.array([0.3,unary_operators_prob,0.5]))
    tree = []
    if leaf_type == 0: #leaf
        key, new_key = jrandom.split(key)
        return sample_readout_leaf(new_key, expressions, sd=3)
    elif leaf_type == 1: #unary operator
        key, new_key1, new_key2 = jrandom.split(key, 3)
        tree.append(expressions.unary_operators[jrandom.randint(new_key1, shape=(), minval=0, maxval=len(expressions.unary_operators))])
        tree.append(sample_readout_tree(new_key2, expressions, depth-1))
        return tree
    elif leaf_type == 2: #binary operator
        key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
        tree.append(expressions.binary_operators[jrandom.randint(new_key1, shape=(), minval=0, maxval=len(expressions.binary_operators))])
        tree.append(sample_readout_tree(new_key2, expressions, depth-1))
        tree.append(sample_readout_tree(new_key3, expressions, depth-1))
        return tree

def sample_trees(key, expressions: Expression, state_size: int, max_depth: int, N: int = 1, num_populations: int = 1, init_method: str = "ramped"):
    "Samples multiple populations of models with a certain (max) depth given a specified method"
    assert (init_method=="ramped") or (init_method=="full") or (init_method=="grow"), "This method is not implemented"

    populations = []
    if init_method=="grow": #Allows for leaves at lower depths
        for _ in range(num_populations):
            population = []
            while len(population) < N:
                trees = []
                for _ in range(state_size):
                    key, new_key1, new_key2 = jrandom.split(key, 3)
                    depth = jrandom.randint(new_key1, (), 2, max_depth+1)
                    trees.append(grow_node(new_key2, expressions, depth))
                
                key, new_key = jrandom.split(key)
                readout = sample_readout_tree(new_key, expressions, max_depth) #Sample readout
                
                new_individual = networkTrees.NetworkTrees(trees, readout)
                if new_individual not in population: #Add individual to population if it is not added yet
                    population.append(new_individual)
            populations.append(population) #Add subpopulation

    elif init_method=="full": #Does not allow for leaves at lower depths
        for _ in range(num_populations):
            population = []
            while len(population) < N:
                trees = []
                for _ in range(state_size):
                    key, new_key1, new_key2 = jrandom.split(key, 3)
                    depth = jrandom.randint(new_key1, (), 2, max_depth+1)
                    trees.append(full_node(new_key2, expressions, depth))

                key, new_key = jrandom.split(key)
                readout = sample_readout_tree(new_key, expressions, max_depth) #Sample readout

                new_individual = networkTrees.NetworkTrees(trees, readout)
                if new_individual not in population: #Add individual to population if it is not added yet
                    population.append(new_individual)
            populations.append(population) #Add subpopulation

    elif init_method=="ramped": #Mixes full and grow initialization, as well as different max depths
        for _ in range(num_populations):
            population = []
            while len(population) < N:
                trees = []
                for _ in range(state_size):
                    key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
                    depth = jrandom.randint(new_key1, (), 2, max_depth+1)                
                    if jrandom.uniform(new_key2)>0.7:
                        trees.append(full_node(new_key3, expressions, depth))
                    else:
                        trees.append(grow_node(new_key3, expressions, depth))

                key, new_key = jrandom.split(key)
                readout = sample_readout_tree(new_key, expressions, max_depth) #Sample readout
                
                new_individual = networkTrees.NetworkTrees(trees, readout)
                if (new_individual not in population): #Add individual to population if it is not added yet
                    population.append(new_individual)
            populations.append(population) #Add subpopulation

    if N == 1: #Return only one tree
        return populations[0][0]      
    return populations
