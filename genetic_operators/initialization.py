import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from expression import Expression
from networks.tree_policy import TreePolicy

def sample_leaf(key: PRNGKey, expressions: Expression, sd: float = 1.0) -> list:
    """Samples a random leaf. The leaf is either a contant or a variable.

    :param key: Random key. 
    :param expressions: Expressions for each layer in a tree policy.
    :param sd: Standard deviation to sample constants.
    :returns: Leaf node.
    """
    key1, key2 = jrandom.split(key)
    
    if jrandom.uniform(key1) < 0.5:
        index = jrandom.randint(key2, shape=(), minval=0, maxval= len(expressions.leaf_nodes))
        return [expressions.leaf_nodes[index]]
    else:
        return [sd * jrandom.normal(key2)]
    

def grow_node(key: PRNGKey, expressions: Expression, depth: int, leaf_sd: float) -> list:
    """Generates a random node that can contain leaves at lower depths.

    :param key: Random key. 
    :param expressions: Expressions for each layer in a tree policy.
    :param depth: Remaining depth of the tree.
    :param leaf_sd: Standard deviation to sample constants.
    :returns: A subtree.
    """
    if depth == 1: #If depth is reached, a leaf is sampled
        key, new_key = jrandom.split(key)
        return sample_leaf(new_key, expressions, sd=leaf_sd)
    
    key, key2 = jrandom.split(key)
    tree = []
    if jrandom.uniform(key) < 0.1:
        return sample_leaf(key2, expressions, sd=leaf_sd)
    else: #operator
        new_key1, new_key2, new_key3 = jrandom.split(key, 3)
        operator = expressions.operators[jrandom.choice(new_key1, a = jnp.arange(len(expressions.operators)), shape=(), p = expressions.operators_prob)]
        tree.append(operator)
        if operator.arity == 2:
            tree.append(grow_node(new_key2, expressions, depth-1, leaf_sd))
            tree.append(grow_node(new_key3, expressions, depth-1, leaf_sd))
        elif operator.arity == 1:
            tree.append(grow_node(new_key2, expressions, depth-1, leaf_sd))
        
        return tree
    
def full_node(key: PRNGKey, expressions: Expression, depth: int, leaf_sd):
    """Generates a random node that can only have leaves at the highest depth.

    :param key: Random key. 
    :param expressions: Expressions for each layer in a tree policy.
    :param depth: Remaing depth of the tree.
    :param leaf_sd: Standard deviation to sample constants.
    :returns: A subtree.
    """
    if depth == 1: #If depth is reached, a leaf is sampled
        return sample_leaf(key, expressions, sd=leaf_sd)
    
    key, new_key = jrandom.split(key)
    tree = []
    new_key1, new_key2, new_key3 = jrandom.split(key, 3)
    operator = expressions.operators[jrandom.choice(new_key1, a = jnp.arange(len(expressions.operators)), shape=(), p = expressions.operators_prob)]
    tree.append(operator)
    if operator.arity == 2:
        tree.append(grow_node(new_key2, expressions, depth-1, leaf_sd))
        tree.append(grow_node(new_key3, expressions, depth-1, leaf_sd))
    elif operator.arity == 1:
        tree.append(grow_node(new_key2, expressions, depth-1, leaf_sd))
    return tree

def sample_trees(key: PRNGKey, expressions: list, layer_sizes: Array, N: int = 1, max_depth: int = 5, init_method: str = "ramped", leaf_sd: float = 1) -> list:
    """Samples tree policies until the population size has been reached.

    :param key: Random key. 
    :param expressions: Expressions for each layer in a tree policy.
    :param layer_sizes: Size of each layer in a tree policy.
    :param N: Number of trees to sample.
    :param max_depth: Highest depth of trees during sampling.
    :param init_method: Method for initializing the trees.
    :param leaf_sd: Standard deviation to sample constants.
    :returns: Leaf node.
    """
    assert (init_method=="ramped") or (init_method=="full") or (init_method=="grow"), "This method is not implemented"
    population = []
    while len(population) < N:
        individual = []
        for i in range(layer_sizes.shape[0]):
            layer = []
            while len(layer) < layer_sizes[i]:
                key, new_key1, new_key2, new_key3 = jrandom.split(key, 4)
                depth = jrandom.randint(new_key1, (), 2, max_depth+1) #Sample depth of tree between 2 and specified max depth
                if init_method=="grow":
                    tree = grow_node(new_key2, expressions[i], depth, leaf_sd)
                    if expressions[i].condition(tree):
                        layer.append(tree)
                elif init_method=="full":
                    tree = full_node(new_key2, expressions[i], depth, leaf_sd)
                    if expressions[i].condition(tree):
                        layer.append(tree)
                elif init_method=="ramped":
                    if jrandom.uniform(new_key3)>0.7: #Sample method to grow tree
                        tree = full_node(new_key2, expressions[i], depth, leaf_sd)
                    else:
                        tree = grow_node(new_key2, expressions[i], depth, leaf_sd)
                    if expressions[i].condition(tree):
                        layer.append(tree)
            individual.append(layer)
        population.append(TreePolicy(individual))

    if N == 1: #Return only one tree
        return population[0]  
    return population