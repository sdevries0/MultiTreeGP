import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from jax import Array
import equinox as eqx
from typing import Tuple
from networks.tree_policy import TreePolicy
from expression import Expression

def index_loc(tree: list, path: list) -> list:
    """Returns subtree at location specified by indices.

    :param tree: Tree. 
    :param path: Path to node in tree.
    :returns: Subtree.
    """
    new_tree = tree
    for i in path:
        new_tree = new_tree[i]
    return new_tree

def key_loc(tree: list, path: list) -> list:
    "Returns subtree at location specified by a path"
    """.

    :param tree: Tree. 
    :param path: Path to node in tree.
    :returns: Subtree.
    """
    new_tree = tree
    for k in path:
        new_tree = new_tree[k.idx]
    return new_tree

def tree_intersection(tree_a: list, tree_b: list, path: list = [], interior_nodes: list = [], boundary_nodes: list = []) -> Tuple[list, list]:
    """Determines the intersecting nodes of a pair of trees. Specifies interior nodes with same arity and boundary nodes with different arity.

    :param tree_a: First parent tree.
    :param tree_b: Second parent tree.
    :param path: Current path to subtree.
    :param interior_nodes: List of interior nodes shared by both trees.
    :param boundary nodes: List of boundary nodes shared by both trees.
    :returns: Shared interior and bounday nodes.
    """
    if (len(tree_a) == len(tree_b)) and len(tree_a) > 1: #Check if same arity but not a leaf
        interior_nodes.append(path + [0])
        if len(tree_b) == 3:
            interior_nodes, boundary_nodes = tree_intersection(tree_a[1], tree_b[1], path + [1], interior_nodes, boundary_nodes)
            interior_nodes, boundary_nodes = tree_intersection(tree_a[2], tree_b[2], path + [2], interior_nodes, boundary_nodes)
        else:
            interior_nodes, boundary_nodes = tree_intersection(tree_a[1], tree_b[1], path + [1], interior_nodes, boundary_nodes)
    else:
        boundary_nodes.append(path + [0])
    
    return interior_nodes, boundary_nodes

def get_subtree(tree: list, key: PRNGKey, expressions: Expression) -> Tuple[list, list]:
    """Return the subtree from a random node onwards.

    :param tree: Tree.
    :param key: Random key. 
    :param expressions: Expressions for each layer in a tree policy.
    :returns: Path to a subtree, and the subtree.
    """
    nodes = jax.tree_util.tree_leaves(tree)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    distribution = jnp.ones(len(nodes)) #increase selection probability of operators
    leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([node not in expressions.operators for node in nodes])))
    for idx in leaf_indices:
        distribution = distribution.at[idx].set(0.1)
    distribution = distribution.at[0].set(0.1) #lower probability of root node
    index = jrandom.choice(key, jnp.arange(len(nodes)),p=distribution)
    path = flat_tree_and_path[index][0][:-1]

    subtree = key_loc(tree, path)

    return path, subtree

def standard_cross_over(trees_a: TreePolicy, trees_b: TreePolicy, reproduction_probability: Array, layer_sizes: Array, key: PRNGKey, expressions: list) -> Tuple[TreePolicy, TreePolicy]:
    """Performs standard cross-over on a pair of trees, returning two new trees. A cross-over point is selected in each pair of trees, interchanging the subtrees below this point.

    :param trees_a: First parent tree policy.
    :param trees_b: Second parent tree policy.
    :param reproduction_probability: Probability of a tree to be adapted in a tree policy.
    :param layer_sizes: Size of each layer in a tree policy.
    :param key: Random key. 
    :param expressions: Expressions for each layer in a tree policy.
    :returns: Offspring tree policies.
    """
    key, new_key = jrandom.split(key)
    #Sample which pair of trees are selected for cross-over. Multiple pairs possible
    crossover_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(jnp.sum(layer_sizes),))
    while jnp.sum(crossover_bool)==0: #Make sure that at least one tree is mutated
        key, new_key = jrandom.split(key)
        crossover_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(jnp.sum(layer_sizes),))
        
    for i in range(layer_sizes.shape[0]):
        for j in range(layer_sizes[i]):
            if i > 0:
                index = jnp.sum(layer_sizes[:i]) + j
            else:
                index = j
            if crossover_bool[index]:
                #Apply cross-over to a pair of trees
                key, key_a, key_b = jrandom.split(key, 3)
                path_a, subtree_a = get_subtree(trees_a()[i][j], key_a, expressions[i])
                path_b, subtree_b = get_subtree(trees_b()[i][j], key_b, expressions[i])

                trees_a = eqx.tree_at(lambda t: key_loc(t()[i][j], path_a), trees_a, subtree_b)
                trees_b = eqx.tree_at(lambda t: key_loc(t()[i][j], path_b), trees_b, subtree_a)

    return trees_a, trees_b

def tree_cross_over(trees_a: TreePolicy, trees_b: TreePolicy, reproduction_probability: Array, layer_sizes: list, key: PRNGKey):
    """Applies cross-over on tree level, interchanging only full trees.

    :param trees_a: First parent tree policy.
    :param trees_b: Second parent tree policy.
    :param reproduction_probability: Probability of a tree to be adapted in a tree policy.
    :param layer_sizes: Size of each layer in a tree policy.
    :param key: Random key. 
    :returns: Offspring tree policies..
    """
    new_trees_a = trees_a
    new_trees_b = trees_b

    key, new_key = jrandom.split(key)
    #Sample which pair of trees are selected for cross-over. Multiple pairs possible
    crossover_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(jnp.sum(layer_sizes),))
    while jnp.sum(crossover_bool)==0: #Make sure that at least one tree is mutated
        key, new_key = jrandom.split(key)
        crossover_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(jnp.sum(layer_sizes),))
    
    for i in range(layer_sizes.shape[0]):
        for j in range(layer_sizes[i]):
            if i > 0:
                index = jnp.sum(layer_sizes[:i]) + j
            else:
                index = j
            if crossover_bool[index]:
            #Apply cross-over to a pair of trees
                new_trees_a = eqx.tree_at(lambda t: t()[i][j], new_trees_a, trees_b()[i][j])
                new_trees_b = eqx.tree_at(lambda t: t()[i][j], new_trees_b, trees_a()[i][j])
    return new_trees_a, new_trees_b

def uniform_cross_over(trees_a: TreePolicy, trees_b: TreePolicy, reproduction_probability: float, layer_sizes: list, key: PRNGKey):
    """Performs uniform cross-over on a pair of trees, returning two new trees. Each overlapping node is switched with 50% chance and children of boundary nodes are switched as well.

    :param trees_a: First parent tree policy.
    :param trees_b: Second parent tree policy.
    :param reproduction_probability: Probability of a tree to be adapted in a tree policy.
    :param layer_sizes: Size of each layer in a tree policy.
    :param key: Random key. 
    :returns: Offspring tree policies.
    """
    
    key, new_key = jrandom.split(key)
    #Sample which pair of trees are selected for cross-over. Multiple pairs possible
    crossover_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(sum(layer_sizes),))
    while jnp.sum(crossover_bool)==0: #Make sure that at least one tree is mutated
        key, new_key = jrandom.split(key)
        crossover_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(sum(layer_sizes),))

    for i in range(layer_sizes.shape[0]):
        for j in range(layer_sizes[i]):
            if i > 0:
                index = jnp.sum(layer_sizes[:i]) + j
            else:
                index = j
            if crossover_bool[index]:
            #Apply cross-over to a pair of trees
                interior_nodes, boundary_nodes = tree_intersection(trees_a()[i][j], trees_b()[i][j], path = [], interior_nodes = [], boundary_nodes = [])
                new_trees_a = trees_a
                new_trees_b = trees_b

                for node in interior_nodes: #Randomly switch two nodes of interior intersecting nodes
                    key, new_key = jrandom.split(key)
                    if jrandom.uniform(new_key) > 0.5:
                        new_trees_a = eqx.tree_at(lambda t: index_loc(t()[i][j], node), new_trees_a, index_loc(trees_b()[i][j], node))
                        new_trees_b = eqx.tree_at(lambda t: index_loc(t()[i][j], node), new_trees_b, index_loc(trees_a()[i][j], node))

                for node in boundary_nodes: #Randomly switch two nodes and their children of boundary intersecting nodes
                    key, new_key = jrandom.split(key)
                    if jrandom.uniform(new_key) > 0.5:
                        new_trees_a = eqx.tree_at(lambda t: index_loc(t()[i][j], node[:-1]), new_trees_a, index_loc(trees_b()[i][j], node[:-1]))
                        new_trees_b = eqx.tree_at(lambda t: index_loc(t()[i][j], node[:-1]), new_trees_b, index_loc(trees_a()[i][j], node[:-1]))
                
                trees_a = new_trees_a
                trees_b = new_trees_b

    return trees_a, trees_b