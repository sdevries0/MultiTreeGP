import jax
import jax.numpy as jnp
import jax.random as jrandom
from miscellaneous.expression import Expression
import miscellaneous.networks as networks

def sample_leaf(key, expressions: Expression, sd: float = 1.0):
    "Samples a random leaf. The leaf is either a contant or a variable"
    index = jrandom.randint(key, shape=(), minval=0, maxval= len(expressions.leaf_nodes))
    return [expressions.leaf_nodes[index]]

def grow_node(key, expressions: Expression, depth: int, unary_operators_prob: float):
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
        tree.append(grow_node(new_key2, expressions, depth-1, unary_operators_prob))
        return tree
    elif leaf_type == 2: #binary operator
        key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
        tree.append(expressions.binary_operators[jrandom.randint(new_key1, shape=(), minval=0, maxval=len(expressions.binary_operators))])
        tree.append(grow_node(new_key2, expressions, depth-1, unary_operators_prob))
        tree.append(grow_node(new_key3, expressions, depth-1, unary_operators_prob))
        return tree
    
def full_node(key, expressions: Expression, depth: int, unary_operators_prob: float):
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
        tree.append(full_node(new_key2, expressions, depth-1, unary_operators_prob))
        return tree
    else: #binary operator
        key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
        tree.append(expressions.binary_operators[jrandom.randint(new_key1, shape=(), minval=0, maxval=len(expressions.binary_operators))])
        tree.append(full_node(new_key2, expressions, depth-1, unary_operators_prob))
        tree.append(full_node(new_key3, expressions, depth-1, unary_operators_prob))
        return tree

def sample_trees(key, expressions: Expression, layer_sizes, N: int = 1, max_depth: int = 5, init_method: str = "ramped", unary_operators_prob: int = 0.0):
    assert (init_method=="ramped") or (init_method=="full") or (init_method=="grow"), "This method is not implemented"
    population = []
    while len(population) < N:
        individual = []
        for i in range(layer_sizes.shape[0]):
            layer = []
            while len(layer) < layer_sizes[i]:
                key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
                depth = jrandom.randint(new_key1, (), 2, max_depth+1)
                if init_method=="grow":
                    tree = grow_node(new_key2, expressions[i], depth, unary_operators_prob = unary_operators_prob)
                    if not expressions[i].condition(tree):
                        layer.append(tree)
                elif init_method=="full":
                    tree = full_node(new_key2, expressions[i], depth, unary_operators_prob = unary_operators_prob)
                    if not expressions[i].condition(tree):
                        layer.append(tree)
                elif init_method=="ramped":
                    if jrandom.uniform(new_key3)>0.7:
                        tree = full_node(new_key2, expressions[i], depth, unary_operators_prob = unary_operators_prob)
                    else:
                        tree = grow_node(new_key2, expressions[i], depth, unary_operators_prob = unary_operators_prob)
                    if not expressions[i].condition(tree):
                        layer.append(tree)
            individual.append(layer)
        population.append(networks.NetworkTrees(individual))

    if N == 1: #Return only one tree
        return population[0]  
    return population