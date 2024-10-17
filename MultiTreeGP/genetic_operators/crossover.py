import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple
from jax import Array
from jax.random import PRNGKey

def sample_indices(carry: Tuple[PRNGKey, Array, float]) -> Tuple[PRNGKey, Array, float]:
    """
    Samples indices of the trees in a candidate that will be mutated
    
    :param key
    :param indices: Indices of trees
    :param reproduction_probability: Probability that determines how likely a tree is mutated

    Returns: The indices of the trees to be mutated
    """

    key, indices, reproduction_probability = carry
    indices = jr.bernoulli(key, p=reproduction_probability, shape=indices.shape)*1.0
    return (jr.split(key, 1)[0], indices, reproduction_probability)

def find_end_idx(carry: Tuple[Array, int, int]) -> Tuple[Array, int, int]:
    """
    Finds the index of the last node in a subtree

    :param tree
    :param open_slots: The number of open slots in the tree that need to be matched with a node
    :param counter: The index of the current node

    Returns: The index of the last node in the subtree
    """

    tree, open_slots, counter = carry
    _, idx1, idx2, _ = tree[counter]
    open_slots -= 1 #Reduce open slot for current node
    open_slots = jax.lax.select(idx1 < 0, open_slots, open_slots+1) #Increase the open slots for a child
    open_slots = jax.lax.select(idx2 < 0, open_slots, open_slots+1) #Increase the open slots for a child
    counter -= 1
    return (tree, open_slots, counter)

def check_equal_subtrees(i, 
                         carry: Tuple[Array, Array, bool]) -> Tuple[Array, Array, bool]:
    """
    Checks if the same node in two subtrees has the same value

    :param tree1
    :param tree2
    :param bool: Indicates whether the previous subtrees are equal
    
    Returns: If the two nodes are equal. If a previous pair of nodes in the subtrees was different, this function will return False
    """
    tree1, tree2, equal = carry

    same_leaf = (tree1[i,3]==tree2[i,3]) & (tree1[i,0]==1)
    equal = jax.lax.select(((tree1[i,0]==tree2[i,0]) & (tree1[i,0]>1)) | same_leaf, equal, False)

    return tree1, tree2, equal

def check_invalid_cx_nodes(carry: Tuple[Array, Array, Array, int, int, Array, Array]) -> bool:
    """
    Checks if the sampled subtrees are different and if the trees after crossover are valid 
    
    :param tree1
    :param tree
    :param node_idx1: Index of node in the first tree
    :param node_idx2: Index of node in the second tree
    
    Returns: If the sampled nodes are valid nodes for crossover
    """

    tree1, tree2, _, node_idx1, node_idx2, _, _ = carry

    _, _, end_idx1 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree1, 1, node_idx1))
    _, _, end_idx2 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree2, 1, node_idx2))

    subtree_size1 = node_idx1 - end_idx1
    subtree_size2 = node_idx2 - end_idx2

    empty_nodes1 = jnp.sum(tree1[:,0]==0)
    empty_nodes2 = jnp.sum(tree2[:,0]==0)

    #If the subtrees have different sizes, they are not equal. If the subtrees have the same size, the nodes in the subtrees are checked to be different
    equal_subtrees = jax.lax.select((subtree_size1==subtree_size2) & ((jnp.sum(tree1[:,0]!=0) > 1) | (jnp.sum(tree2[:,0]!=0) > 1)), 
                                    jax.lax.fori_loop(0, subtree_size1, check_equal_subtrees, (jnp.roll(tree1, -node_idx1 + subtree_size1 - 1, axis=0), 
                                                                                               jnp.roll(tree2, -node_idx2 + subtree_size2 - 1, axis=0), 
                                                                                               True))[2], 
                                    False)

    #Check if the subtrees can be inserted
    return (empty_nodes1 < subtree_size2 - subtree_size1) | (empty_nodes2 < subtree_size1 - subtree_size2) | equal_subtrees

def sample_cx_nodes(carry: Tuple[Array, Array, Array, int, int, Array, Array]) -> Tuple[Array, Array, Array, int, int, Array, Array]:
    """
    Samples nodes in a pair of trees for crossover
    
    :param tree1
    :param tree
    :param keys
    :param node_ids: Indices of all the nodes in the trees
    :param operator_indices: The indices that belong to operator nodes
    
    Returns: Sampled nodes
    """

    tree1, tree2, keys, _, _, node_ids, operator_indices = carry
    key1, key2 = keys

    #Sample nodes from the non-empty nodes, with higher probability for operator nodes
    cx_prob1 = jnp.isin(tree1[:,0], operator_indices)
    cx_prob1 = jnp.where(tree1[:,0]==0, cx_prob1, cx_prob1+1)
    node_idx1 = jr.choice(key1, node_ids, p = cx_prob1*1.)

    cx_prob2 = jnp.isin(tree2[:,0], operator_indices)
    cx_prob2 = jnp.where(tree2[:,0]==0, cx_prob2, cx_prob2+1)
    node_idx2 = jr.choice(key2, node_ids, p = cx_prob2*1.)

    return (tree1, tree2, jr.split(key1), node_idx1, node_idx2, node_ids, operator_indices)

def crossover(tree1: Array, 
              tree2: Array, 
              keys: Array,
              max_nodes: int, 
              operator_indices: Array) -> Tuple[Array, Array]:
    """
    Applies crossover to a pair of trees to produce two new trees
    
    :param tree1
    :param tree2
    :param keys
    :param max_nodes: Max number of nodes in a tree
    :param operator_indices: The indices that belong to operator nodes
    
    Returns: Pair of new trees
    """

    #Define indices of the nodes
    node_ids = jnp.arange(max_nodes)
    tree_indices = jnp.tile(node_ids[:,None], reps=(1,4))
    key1, key2 = keys

    #Define last node in tree
    last_node_idx1 = jnp.sum(tree1[:,0]==0)
    last_node_idx2 = jnp.sum(tree2[:,0]==0)

    #Randomly select nodes for crossover
    _, _, _, node_idx1, node_idx2, _, _ = sample_cx_nodes((tree1, tree2, jr.split(key1), 0, 0, node_ids, operator_indices))

    #Reselect until valid crossover nodes have been found
    _, _, _, node_idx1, node_idx2, _, _ = jax.lax.while_loop(check_invalid_cx_nodes, sample_cx_nodes, (tree1, tree2, jr.split(key2), node_idx1, node_idx2, node_ids, operator_indices))

    #Retrieve subtrees of selected nodes
    _, _, end_idx1 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree1, 1, node_idx1))
    _, _, end_idx2 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree2, 1, node_idx2))

    #Initialize children
    child1 = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child2 = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))

    #Compute subtree sizes
    subtree_size1 = node_idx1 - end_idx1
    subtree_size2 = node_idx2 - end_idx2

    #Insert nodes before subtree in children
    child1 = jnp.where(tree_indices >= node_idx1 + 1, tree1, child1)
    child2 = jnp.where(tree_indices >= node_idx2 + 1, tree2, child2)
    
    #Align nodes after subtree with first open spot after new subtree in children
    rolled_tree1 = jnp.roll(tree1, subtree_size1 - subtree_size2, axis=0)
    rolled_tree2 = jnp.roll(tree2, subtree_size2 - subtree_size1, axis=0)

    #Insert nodes after subtree in children
    child1 = jnp.where((tree_indices >= node_idx1 - subtree_size2 - (end_idx1 - last_node_idx1)) & (tree_indices < node_idx1 + 1 - subtree_size2), rolled_tree1, child1)
    child2 = jnp.where((tree_indices >= node_idx2 - subtree_size1 - (end_idx2 - last_node_idx2)) & (tree_indices < node_idx2 + 1 - subtree_size1), rolled_tree2, child2)

    #Update index references to moved nodes in staying nodes
    child1 = child1.at[:,1:3].set(jnp.where((child1[:,1:3] < (node_idx1 - subtree_size1 + 1)) & (child1[:,1:3] > -1), child1[:,1:3] + (subtree_size1-subtree_size2), child1[:,1:3]))
    child2 = child2.at[:,1:3].set(jnp.where((child2[:,1:3] < (node_idx2 - subtree_size2 + 1)) & (child2[:,1:3] > -1), child2[:,1:3] + (subtree_size2-subtree_size1), child2[:,1:3]))

    #Align subtree with the selected node in children
    rolled_subtree1 = jnp.roll(tree1, node_idx2 - node_idx1, axis=0)
    rolled_subtree2 = jnp.roll(tree2, node_idx1 - node_idx2, axis=0)

    #Update index references in subtree
    rolled_subtree1 = rolled_subtree1.at[:,1:3].set(jnp.where(rolled_subtree1[:,1:3] > -1, rolled_subtree1[:,1:3] + (node_idx2 - node_idx1), -1))
    rolled_subtree2 = rolled_subtree2.at[:,1:3].set(jnp.where(rolled_subtree2[:,1:3] > -1, rolled_subtree2[:,1:3] + (node_idx1 - node_idx2), -1))

    #Insert subtree in selected node in children
    child1 = jnp.where((tree_indices >= node_idx1 + 1 - subtree_size2) & (tree_indices < node_idx1 + 1), rolled_subtree2, child1)
    child2 = jnp.where((tree_indices >= node_idx2 + 1 - subtree_size1) & (tree_indices < node_idx2 + 1), rolled_subtree1, child2)
    
    return child1, child2

def crossover_trees(parent1: Array, 
                    parent2: Array, 
                    keys: Array, 
                    reproduction_probability: float, 
                    max_nodes: int, 
                    operator_indices: Array) -> Tuple[Array, Array]:
    """
    Applies crossover to the trees in a pair of candidates

    :param parent1
    :param parent2
    :param keys
    :param reproduction_probability: Probability of a tree to be mutated
    :param max_nodes: Max number of nodes in a tree
    :param operator_indices: The indices that belong to operator nodes 
    
    Returns: Pair of candidates after crossover    
    """

    #Determine to which trees in the candidates crossover is applied
    _, cx_indices, _ = jax.lax.while_loop(lambda carry: jnp.sum(carry[1])==0, sample_indices, (keys[0, 0], jnp.zeros(parent1.shape[0]), reproduction_probability))
    offspring1, offspring2 = jax.vmap(crossover, in_axes=[0,0,0,None,None])(parent1, parent2, keys, max_nodes, operator_indices)
    child1 = jnp.where(cx_indices[:,None,None] * jnp.ones_like(parent1), offspring1, parent1)
    child2 = jnp.where(cx_indices[:,None,None] * jnp.ones_like(parent2), offspring2, parent2)
    return child1, child2