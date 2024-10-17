import jax
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
from functools import partial
from jax import Array
from typing import Tuple

def sample_node(i: int, 
                carry: Tuple[PRNGKey, Array, int, int, int, Array, Tuple]):
    """
    Samples nodes sequentially. The order of the sampled nodes is in breadth first, but nodes are stored depth-first

    :param key
    :param tree
    :param open_slots: Node positions that have to be filled
    :param max_init_depth: Max depth in a tree at initialization
    :param max_nodes: Max number of nodes in a tree
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for initialization
    
    Returns: Tree with sampled node inserted at the given index
    """

    key, tree, open_slots, max_init_depth, max_nodes, variable_array, args = carry
    variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd, map_b_to_d = args
    coefficient_key, leaf_key, variable_key, node_key, operator_key = jr.split(key, 5)
    _i = map_b_to_d[i].astype(int) #Get depth first index

    depth = (jnp.log(i+1)/jnp.log(2)).astype(int) #Compute depth of node
    coefficient = jr.normal(coefficient_key)*coefficient_sd
    leaf = jax.lax.select(jr.uniform(leaf_key)<0.5, 1, jr.choice(variable_key, variable_indices, shape=(), p=variable_array)) #Sample coefficient or variable

    index = jax.lax.select((open_slots < max_nodes - i - 1) & (depth+1<max_init_depth), #Check if max depth has been reached, or if the number of open slots reached the max number of nodes
                            jax.lax.select(jr.uniform(node_key)<(0.7**depth), #At higher depth, a leaf node is more probable
                                            jr.choice(operator_key, a=operator_indices, shape=(), p=operator_probabilities), 
                                            leaf), 
                            leaf)
    
    index = jax.lax.select(open_slots == 0, 0, index) #If there are no open slots, the node should be empty

    #If parent node is a leaf, the node should be empty
    index = jax.lax.select(i>0, jax.lax.select((slots[jnp.maximum(tree[map_b_to_d[(i + (i%2) - 2)//2].astype(int), 0], 0).astype(int)] + i%2) > 1, index, 0), index)

    #Set index references
    tree = jax.lax.select(slots[index] > 0, tree.at[_i, 1].set(map_b_to_d[2*i+1]), tree.at[_i, 1].set(-1))
    tree = jax.lax.select(slots[index] > 1, tree.at[_i, 2].set(map_b_to_d[2*i+2]), tree.at[_i, 2].set(-1))

    tree = jax.lax.select(index == 1, tree.at[_i,3].set(coefficient), tree) #Set coefficient value
    tree = tree.at[_i, 0].set(index)

    open_slots = jax.lax.select(index == 0, open_slots, jnp.maximum(0, open_slots + slots[index] - 1)) #Update the number of open slots

    return (jr.fold_in(key, i), tree, open_slots, max_init_depth, max_nodes, variable_array, args)

def prune_row(i: int, 
              carry: Tuple[Array, int, int], 
              old_tree: Array) -> Array:
    """
    Sequentially adds nodes to the new tree if it is not empty
    
    :param tree
    :param counter: Counter for the next node location in the new tree
    :param tree_size: Max size of the tree
    :param old_tree: Tree with empty nodes that have to be pruned

    Returns: Tree with empty nodes pruned    
    """

    tree, counter, tree_size = carry

    _i = tree_size - i - 1

    row = old_tree[_i]

    #If node is not empty, add node and update index references
    tree = jax.lax.select(row[0] != 0, tree.at[counter].set(row), tree.at[:,1:3].set(jnp.where(tree[:,1:3] > _i, tree[:,1:3]-1, tree[:,1:3])))
    counter = jax.lax.select(row[0] != 0, counter - 1, counter)

    return (tree, counter, tree_size)
    
def prune_tree(tree: Array, 
               tree_size: int, 
               max_nodes: int) -> Array:
    """
    Removes empty nodes from a tree. The new tree is filled with empty nodes at the end to match the max number of nodes
    
    :param tree
    :param tree_size: Max size of the old tree
    :param max_nodes: Max number of nodes in the new tree
    :param old_tree: Tree with empty nodes that have to be pruned

    Returns: Tree with empty nodes pruned    
    """

    tree, counter, _ = jax.lax.fori_loop(0, tree_size, partial(prune_row, old_tree=tree), (jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1)), max_nodes-1, tree_size))
    tree = tree.at[:,1:3].set(jnp.where(tree[:,1:3]>-1, tree[:,1:3] + counter + 1, tree[:,1:3])) #Update index references after pruning
    return tree

def sample_tree(key: PRNGKey, 
                depth: int, 
                variable_array: Array, 
                max_init_depth: int, 
                max_nodes: int, 
                args: Tuple) -> Array:
    """
    Initializes a tree

    :param key
    :param max_init_depth: Max depth in a tree at initialization
    :param max_nodes: Max number of nodes in a tree
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for initialization
    
    Returns: Tree
    """

    #First sample tree at full size given depth
    tree_size = 2**max_init_depth-1
    tree = jax.lax.fori_loop(0, tree_size, sample_node, (key, jnp.zeros((tree_size, 4)), 1, depth, max_nodes, variable_array, args))[1] #Sample nodes in a tree sequentially

    #Prune empty rows in tree
    pruned_tree =  prune_tree(tree, tree_size, max_nodes)
    return pruned_tree

def sample_trees(keys: Array, 
                 max_init_depth: int, 
                 max_nodes: int, 
                 variable_array: Array, 
                 args: Tuple) -> Array:
    """
    Initializes a candidate consisting of multiple trees

    :param keys
    :param max_init_depth: Max depth in a tree at initialization
    :param max_nodes: Max number of nodes in a tree
    :param variable_array: The valid variables for each tree
    :param args: Miscellaneous parameters required for initialization
    
    Returns: Candidate
    """
    return jax.vmap(sample_tree, in_axes=[0, None, 0, None, None, None])(keys, max_init_depth, variable_array, max_init_depth, max_nodes, args)

def sample_population(key: PRNGKey, 
                      population_size: int, 
                      num_trees: int, 
                      max_init_depth: int, 
                      max_nodes: int, 
                      variable_array: Array, 
                      args: Tuple) -> Array:
    """
    Initializes a population of candidates

    :param key
    :param population_size: Number of candidates that have to be sampled
    :param num_trees: Number of trees in a candidate
    :param max_init_depth: Max depth in a tree at initialization
    :param max_nodes: Max number of nodes in a tree
    :param variable_array: The valid variables for each tree
    :param args: Miscellaneous parameters required for initialization
    
    Returns: Population of candidates
    """
    return jax.vmap(sample_trees, in_axes=[0, None, None, None, None])(jr.split(key, (population_size, num_trees)), max_init_depth, max_nodes, variable_array, args)
    