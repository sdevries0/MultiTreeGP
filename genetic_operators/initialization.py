import jax
import jax.numpy as jnp
import jax.random as jrandom
from miscellaneous.expression import Expression
import miscellaneous.networks as networks

def sample_leaf(key, expressions: Expression, sd: float = 1.0):
    "Samples a random leaf. The leaf is either a contant or a variable"
    key1, key2 = jrandom.split(key)
    
    if jrandom.uniform(key1) < 0.7:
        index = jrandom.randint(key2, shape=(), minval=0, maxval= len(expressions.leaf_nodes))
        return [expressions.leaf_nodes[index]]
    else:
        return [sd * jrandom.normal(key2)]
    

def grow_node(key, expressions: Expression, depth: int):
    "Generates a random node that can contain leaves at lower depths"
    if depth == 1: #If depth is reached, a leaf is sampled
        key, new_key = jrandom.split(key)
        return sample_leaf(new_key, expressions, sd=3)
    
    key, key2 = jrandom.split(key)
    tree = []
    if jrandom.uniform(key) < 0.3:
        return sample_leaf(key2, expressions, sd=3)
    else: #operator
        new_key1, new_key2, new_key3 = jrandom.split(key, 3)
        operator = expressions.operators[jrandom.choice(new_key1, a = jnp.arange(len(expressions.operators)), shape=(), p = expressions.operators_prob)]
        tree.append(operator)
        if operator.arity == 2:
            tree.append(grow_node(new_key2, expressions, depth-1))
            tree.append(grow_node(new_key3, expressions, depth-1))
        elif operator.arity == 1:
            tree.append(grow_node(new_key2, expressions, depth-1))
        
        return tree
    
def full_node(key, expressions: Expression, depth: int):
    "Generates a random node that can only have leaves at the highest depth"
    if depth == 1: #If depth is reached, a leaf is sampled
        return sample_leaf(key, expressions, sd=3)
    
    key, new_key = jrandom.split(key)
    tree = []
    new_key1, new_key2, new_key3 = jrandom.split(key, 3)
    operator = expressions.operators[jrandom.choice(new_key1, a = jnp.arange(len(expressions.operators)), shape=(), p = expressions.operators_prob)]
    tree.append(operator)
    if operator.arity == 2:
        tree.append(grow_node(new_key2, expressions, depth-1))
        tree.append(grow_node(new_key3, expressions, depth-1))
    elif operator.arity == 1:
        tree.append(grow_node(new_key2, expressions, depth-1))
    return tree

def sample_trees(key, expressions: Expression, layer_sizes, N: int = 1, max_depth: int = 5, init_method: str = "ramped"):
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
                    tree = grow_node(new_key2, expressions[i], depth)
                    if not expressions[i].condition(tree):
                        layer.append(tree)
                elif init_method=="full":
                    tree = full_node(new_key2, expressions[i], depth)
                    if not expressions[i].condition(tree):
                        layer.append(tree)
                elif init_method=="ramped":
                    if jrandom.uniform(new_key3)>0.7:
                        tree = full_node(new_key2, expressions[i], depth)
                    else:
                        tree = grow_node(new_key2, expressions[i], depth)
                    if not expressions[i].condition(tree):
                        layer.append(tree)
            individual.append(layer)
        population.append(networks.NetworkTrees(individual))

    if N == 1: #Return only one tree
        return population[0]  
    return population