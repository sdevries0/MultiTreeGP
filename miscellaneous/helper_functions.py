import jax
import jax.numpy as jnp
from typing import Sequence
from miscellaneous.expression import Expression

def index_loc(tree: list, path: list):
    "Returns subtree at location specified by indices"
    new_tree = tree
    for i in path:
        new_tree = new_tree[i]
    return new_tree

def key_loc(tree: list, path: list):
    "Returns subtree at location specified by a path"
    new_tree = tree
    for k in path:
        new_tree = new_tree[k.idx]
    return new_tree

def is_operator(symbol, expressions: Expression):
    "Checks if node is an operator"
    return symbol in expressions.operators

def is_leaf(symbol, expressions: Expression):
    "Checks if node is a leaf"
    return symbol not in expressions.operators

def is_constant(symbol):
    "Checks if node is a constant"
    return isinstance(symbol, jax.numpy.ndarray)
    
def tree_depth(tree: list):
    "Returns highest depth of tree"
    flat_tree = jax.tree_util.tree_leaves_with_path(tree)
    return jnp.max(jnp.array([len(node[0]) for node in flat_tree]))

def depth_per_node(tree: list):
    "Returns depth of each node of the tree"
    flat_tree = jax.tree_util.tree_leaves_with_path(tree)
    return jnp.array([len(node[0]) for node in flat_tree])

def flatten(populations: list):
    #Flattens all subpopulations
    return [candidate for population in populations for candidate in population]

def best_solution(populations: list, fitnesses: Sequence):
    "Returns the best solution in all subpopulations"
    best_solution = None
    best_fitness = jnp.inf
    for pop in range(len(populations)):
        if jnp.min(fitnesses[pop]) < best_fitness:
            best_fitness = jnp.min(fitnesses[pop])
            best_solution = populations[pop][jnp.argmin(fitnesses[pop])]
    return best_solution