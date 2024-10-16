import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from typing import Tuple, Callable
from jax import Array
from jax.random import PRNGKey

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

def sample_leaf_node(carry: Tuple[Array, PRNGKey, int, int, float, Array]) -> Tuple[Array, PRNGKey, int, int, float, Array]:
    """
    Samples a leaf node to be replaced in the tree and a new leaf node
    
    :param tree
    :param key
    :param variable_array: The valid variables for this tree
    :param variable_indices: The indices of the available leaf nodes

    Returns: The index of the leaf node that will be replaced with a new leaf
    """

    tree, key, _, _, variable_array, variable_indices = carry
    key, select_key, sample_key, variable_key = jr.split(key, 4)
    node_ids = tree[:,0]
    is_leaf = (node_ids == 1) | jnp.isin(node_ids, variable_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.) #Sample node to be mutated
    new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, 1, jr.choice(variable_key, variable_indices, shape=(), p=variable_array)) #Sample coefficient or variable

    return (tree, key, mutate_idx, new_leaf, variable_array, variable_indices)

def check_equal_leaves(carry: Tuple[Array, PRNGKey, int, int, float, Array]) -> bool:
    """
    Checks that the old and new leaf node are different
    
    :param tree
    :param key
    :param mutate_idx: Index of the node that is mutated
    :param new_leaf: Value of the new leaf node

    Returns: Whether the old and new leaf are different
    """

    tree, key, mutate_idx, new_leaf, _, _ = carry

    return (tree[mutate_idx, 0] == new_leaf) & (new_leaf != 1) #Nodes have to be different unless they are coefficients

def check_invalid_operator_node(carry: Tuple[Array, PRNGKey, int, int, int, Array, Array]) -> bool:
    """
    Checks that the old and new operator node are different and that the tree does not exceed the maximum size after sampling a new subtree
    
    :param tree
    :param key
    :param mutate_idx: Index of the node that is mutated
    :param new_operator: Value of the new operator node
    :param slots: The number of operands of the operators

    Returns: Whether the old and new operator are different and a valid subtree can be sampled
    """

    tree, _, mutate_idx, new_operator, slots, _, _ = carry
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, mutate_idx))

    subtree_size = mutate_idx - end_idx

    empty_nodes = jnp.sum(tree[:,0]==0)
    new_tree_size = jax.lax.select(slots[new_operator] == 2, 7, 8) #The maximum size of the subtree given the number of operands of the new operator

    #Nodes have to be different and the new tree size should be smaller than the number of empty nodes and removed subtree size
    return (tree[mutate_idx, 0] == new_operator) | (empty_nodes + subtree_size < new_tree_size)

def sample_operator_node(carry: Tuple[Array, PRNGKey, int, int, int, Array, Array]) -> Tuple[Array, PRNGKey, int, int, int, Array, Array]:
    """
    Samples an operator node to be replaced in the tree and a new operator node
    
    :param tree
    :param key
    :param slots: The valid variables for this tree
    :param operator_indices: The indices of the available operator nodes
    :param operator_probabilities: The probabilities of the operator nodes

    Returns: The index of the operator node that will be replaced with a new operator
    """

    tree, key, _, _, slots, operator_indices, operator_probabilities = carry
    key, select_key, sample_key = jr.split(key, 3)

    node_ids = tree[:,0]
    is_operator = jnp.isin(node_ids, operator_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.) #Sample node to be mutated
    new_operator = jr.choice(sample_key, a=operator_indices, shape=(), p=operator_probabilities) #Sample operator

    return (tree, key, mutate_idx, new_operator, slots, operator_indices, operator_probabilities)

def add_subtree(tree: Array, 
                key: PRNGKey, 
                variable_array: Array, 
                args: Tuple) -> Array:
    """
    Replaces a leaf node with a random subtree
    
    :param tree
    :param key
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Mutated tree
    """

    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    select_key, sample_key = jr.split(key, 2)
    node_ids = tree[:,0]
    is_leaf = (node_ids == 1) | jnp.isin(node_ids, variable_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.) #Sample node to be mutated

    subtree = sample_tree(sample_key, 2, variable_array) #Sample new subtree
    subtree_size = jnp.sum(subtree[:,0]!=0)

    remaining_size = mutate_idx - jnp.sum(tree[:,0]==0) #Size of the subtree that should be preserved in the tree

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child = jnp.where(tree_indices > mutate_idx, tree, child) #Insert nodes before the mutation index in the new tree

    rolled_tree = jnp.roll(tree, -subtree_size + 1, axis=0) #Align position of the remaining nodes with the new tree
    child = jnp.where((tree_indices <= mutate_idx - subtree_size) & (tree_indices > mutate_idx - subtree_size - remaining_size), rolled_tree, child) #Insert nodes after the subtree in the new tree
    child = child.at[:,1:3].set(jnp.where((child[:,1:3] < (mutate_idx)) & (child[:,1:3] > -1), child[:,1:3] - (subtree_size - 1), child[:,1:3])) #Update index references

    subtree = jnp.roll(subtree, -(max_nodes - mutate_idx - 1), axis=0) #Align position of the subtree with the new tree
    subtree = subtree.at[:,1:3].set(jnp.where(subtree[:,1:3] > -1, subtree[:,1:3] + (mutate_idx - max_nodes + 1), -1)) #Insert subtree in the new tree
    child = jnp.where((tree_indices <= mutate_idx) & (tree_indices > mutate_idx - subtree_size), subtree, child) #Update index references in subtree

    return child

def mutate_leaf(tree: Array, 
                key: PRNGKey, 
                variable_array: Array, 
                args: Tuple) -> Array:
    """
    Replaces a leaf node with a different leaf node
    
    :param tree
    :param key
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Mutated tree
    """

    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args
    select_key, sample_key, coefficient_key, variable_key = jr.split(key, 4)
    node_ids = tree[:,0]
    is_leaf = (node_ids == 1) | jnp.isin(node_ids, variable_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.) #Sample node to be mutated

    new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, 1, jr.choice(variable_key, variable_indices, shape=(), p=variable_array)) #Sample coefficient or variable

    #Check that the new leaf is different from the old leaf
    _, _, mutate_idx, new_leaf, _, _ = jax.lax.while_loop(check_equal_leaves, sample_leaf_node, (tree, jr.fold_in(key, 0), mutate_idx, new_leaf, variable_array, variable_indices))

    coefficient = jr.normal(coefficient_key)*coefficient_sd

    child = tree.at[mutate_idx, 0].set(new_leaf)
    child = jax.lax.select(new_leaf==1, child.at[mutate_idx, 3].set(coefficient), child.at[mutate_idx, 3].set(0)) #Set coefficient value

    return child

def replace_with_one_subtree(tree: Array, 
                             key: PRNGKey, 
                             mutate_idx: int, 
                             operator: int, 
                             variable_array: Array,
                             args: Tuple) -> Array:
    """
    Replaces node with a operator node with one operand
    
    :param tree
    :param key
    :param mutate_idx: Index of the node that is mutated
    :param operator: Operator node that is inserted
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Mutated tree
    """

    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args

    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, mutate_idx))

    remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1 #Size of the subtree that should be preserved in the tree

    subtree = sample_tree(key, 2, variable_array) #Sample new subtree
    subtree_size = jnp.sum(subtree[:,0]!=0)

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))

    child = jnp.where(tree_indices >= mutate_idx, tree, child) #Insert nodes before the mutation index in the new tree

    rolled_tree = jnp.roll(tree, (mutate_idx - end_idx - subtree_size - 1), axis=0) #Align position of the remaining nodes with the new tree
    child = jnp.where((tree_indices < mutate_idx - subtree_size) & (tree_indices >= mutate_idx - subtree_size - remaining_size), rolled_tree, child) #Insert nodes after the subtree in the new tree

    child = child.at[mutate_idx, 0].set(operator)
    child = child.at[mutate_idx, 2].set(-1) #Remove index reference to second operand

    child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] + (mutate_idx - end_idx - subtree_size - 1), child[:,1:3])) #Update index references

    subtree = jnp.roll(subtree, -(max_nodes - mutate_idx), axis=0) #Align position of the subtree with the new tree
    subtree = subtree.at[:,1:3].set(jnp.where(subtree[:,1:3] > -1, subtree[:,1:3] + (mutate_idx - max_nodes), -1)) #Update index references in subtree
    child = jnp.where((tree_indices < mutate_idx) & (tree_indices > mutate_idx - subtree_size - 1), subtree, child) #Insert subtree in the new tree

    return child

def replace_with_two_subtrees(tree: Array, 
                              key: PRNGKey, 
                              mutate_idx: int, 
                              operator: int, 
                              variable_array: Array, 
                              args: Tuple) -> Array:
    """
    Replaces node with a operator node with two operands
    
    :param tree
    :param key
    :param mutate_idx: Index of the node that is mutated
    :param operator: Operator node that is inserted
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Mutated tree
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    key1, key2 = jr.split(key)
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, mutate_idx))

    remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1 #Size of the subtree that should be preserved in the tree

    #Sample new subtrees
    subtree1 = sample_tree(key1, 1, variable_array)
    subtree1_size = jnp.sum(subtree1[:,0]!=0)
    subtree2 = sample_tree(key2, 1, variable_array)
    subtree2_size = jnp.sum(subtree2[:,0]!=0)

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child = jnp.where(tree_indices >= mutate_idx, tree, child) #Insert nodes before the mutation index in the new tree

    rolled_tree = jnp.roll(tree, (mutate_idx - end_idx - subtree1_size - subtree2_size - 1), axis=0) #Align position of the remaining nodes with the new tree
    child = jnp.where((tree_indices < mutate_idx - subtree1_size - subtree2_size) & (tree_indices >= mutate_idx - subtree1_size - subtree2_size - remaining_size), rolled_tree, child) #Insert nodes after the subtrees in the new tree

    child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] + (mutate_idx - end_idx - subtree1_size - subtree2_size - 1), child[:,1:3])) #Update index references

    child = child.at[mutate_idx, 0].set(operator)
    child = child.at[mutate_idx, 1].set(mutate_idx - 1)
    child = child.at[mutate_idx, 2].set(mutate_idx - subtree1_size - 1)

    subtree1 = jnp.roll(subtree1, -(max_nodes - mutate_idx), axis=0) #Align position of the first subtree with the new tree
    subtree1 = subtree1.at[:,1:3].set(jnp.where(subtree1[:,1:3] > -1, subtree1[:,1:3] + (mutate_idx - max_nodes), -1)) #Update index references in subtree
    child = jnp.where((tree_indices < mutate_idx) & (tree_indices > mutate_idx - subtree1_size - 1), subtree1, child) #Insert subtree in the new tree

    subtree2 = jnp.roll(subtree2, -(max_nodes - mutate_idx + subtree1_size), axis=0) #Align position of the second subtree with the new tree
    subtree2 = subtree2.at[:,1:3].set(jnp.where(subtree2[:,1:3] > -1, subtree2[:,1:3] + (mutate_idx - subtree1_size - max_nodes), -1)) #Update index references in subtree
    child = jnp.where((tree_indices < mutate_idx - subtree1_size) & (tree_indices > mutate_idx - subtree1_size - subtree2_size - 1), subtree2, child) #Insert subtree in the new tree

    return child

def mutate_operator(tree: Array, 
                    key: PRNGKey, 
                    variable_array: Array, 
                    args: Tuple) -> Array:
    """
    Replaces an operator node with a different operator node. The arity of the operator might change, therefore new subtrees may be sampled
    
    :param tree
    :param key
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Mutated tree
    """

    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args
    select_key, sample_key, subtree_key = jr.split(key, 3)
    node_ids = tree[:,0]
    is_operator = jnp.isin(node_ids, operator_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.) #Sample node to be mutated

    new_operator = jr.choice(sample_key, a=operator_indices, shape=(), p=operator_probabilities) #Sample new operator

    #Check that the new operator is different from the old operator
    _, _, mutate_idx, new_operator, _, _, _ = jax.lax.while_loop(check_invalid_operator_node, sample_operator_node, (tree, 
                                                                                                                     jr.fold_in(key, 0),
                                                                                                                     mutate_idx, 
                                                                                                                     new_operator, 
                                                                                                                     slots, 
                                                                                                                     operator_indices, 
                                                                                                                     operator_probabilities))

    current_slots = slots[node_ids[mutate_idx].astype(int)]
    new_slots = slots[new_operator]

    #Insert new operator and sample subtrees if necessary
    child = jax.lax.select(current_slots==2, 
                            jax.lax.select(new_slots==2, tree.at[mutate_idx, 0].set(new_operator), replace_with_one_subtree(tree, subtree_key, mutate_idx, new_operator, variable_array, args)), 
                            jax.lax.select(new_slots==2, replace_with_two_subtrees(tree, subtree_key, mutate_idx, new_operator, variable_array, args), tree.at[mutate_idx, 0].set(new_operator)))

    return child

def delete_operator(tree: Array, 
                    key: PRNGKey, 
                    variable_array: Array, 
                    args: Tuple) -> Array:
    """
    Replaces an operator and operands with a leaf node
    
    :param tree
    :param key
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Mutated tree
    """

    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    select_key, sample_key, coefficient_key, variable_key = jr.split(key, 4)
    node_ids = tree[:,0]
    is_operator = jnp.isin(node_ids, operator_indices)
    is_operator = is_operator.at[-1].set(False)
    delete_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.) #Sample node to be mutated

    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, delete_idx))

    remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1 #Size of the subtree that should be preserved in the tree

    coefficient = jr.normal(coefficient_key)*coefficient_sd
    new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, 1, jr.choice(variable_key, variable_indices, shape=(), p=variable_array)) #Sample coefficient or variable

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child = jnp.where(tree_indices > delete_idx, tree, child) #Insert nodes before the mutation index in the new tree

    rolled_tree = jnp.roll(tree, delete_idx - end_idx - 1, axis=0) #Align position of the remaining nodes with the new tree
    child = jnp.where((tree_indices < delete_idx) & (tree_indices >= delete_idx - remaining_size), rolled_tree, child) #Insert nodes after the subtrees in the new tree
    child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (delete_idx - 1)) & (child[:,1:3] > -1), child[:,1:3] + (delete_idx - end_idx - 1), child[:,1:3])) #Update index references

    child = child.at[delete_idx, 0].set(new_leaf) #Insert leaf node
    child = jax.lax.select(new_leaf==1, child.at[delete_idx, 3].set(coefficient), child.at[delete_idx, 3].set(0)) #Set coefficient value

    return child

def prepend_operator(tree: Array, 
                     key: PRNGKey, 
                     variable_array: Array, 
                     args: Tuple) -> Array:
    """
    Adds an operator node before root node
    
    :param tree
    :param key
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Mutated tree
    """

    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    sample_key, subtree_key, side_key = jr.split(key, 3)
    new_operator = jr.choice(sample_key, a=operator_indices, shape=(), p=operator_probabilities) #Sample new operator
    new_slots = slots[new_operator]

    subtree = sample_tree(subtree_key, 2, variable_array) #Sample new subtree
    subtree_size = jnp.sum(subtree[:,0]!=0)
    tree_size = jnp.sum(tree[:,0]!=0)

    second_branch = jr.bernoulli(side_key) #Sample if the old tree is the first or second operand

    child = jnp.roll(tree, -1 - (new_slots - 1) * second_branch*subtree_size, axis=0) #Insert old tree in the new tree
    child = child.at[:,1:3].set(jnp.where(child[:,1:3] > -1, child[:,1:3] - 1 - (new_slots - 1) * second_branch*subtree_size, child[:,1:3])) #Update index references

    rolled_subtree = jnp.roll(subtree, -1 - (1-second_branch) * tree_size, axis=0) #Align position of the new subtree with the new tree
    rolled_subtree = rolled_subtree.at[:,1:3].set(jnp.where(rolled_subtree[:,1:3] > -1, rolled_subtree[:,1:3] - 1 - (1-second_branch)*tree_size, rolled_subtree[:,1:3])) #Update index references in subtree

    #Insert subtree in first or second branch of new tree
    child_2_branches = jax.lax.select(second_branch, 
                                      jnp.where((tree_indices < max_nodes - 1) & (tree_indices >= max_nodes - subtree_size - 1), rolled_subtree, child), 
                                      jnp.where((tree_indices < max_nodes - tree_size - 1) & (tree_indices >= max_nodes - tree_size - subtree_size - 1), rolled_subtree, child))

    child = jax.lax.select(new_slots==2, child_2_branches, child) #Select tree with one or two operands
    child = child.at[-1, 0].set(new_operator)
    child = child.at[-1, 1].set(max_nodes - 2)
    child = child.at[-1, 2].set(jax.lax.select(new_slots==2, max_nodes - jax.lax.select(second_branch, subtree_size, tree_size) - 2, -1))

    return child

def insert_operator(tree: Array, 
                    key: PRNGKey, 
                    variable_array: Array, 
                    args: Tuple) -> Array:
    """
    Inserts an operator node above a random node
    
    :param tree
    :param key
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Mutated tree
    """

    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    select_key, sample_key, subtree_key, side_key = jr.split(key, 4)
    node_ids = tree[:,0]
    is_operator = jnp.isin(node_ids, operator_indices)
    is_operator = is_operator.at[-1].set(False) #Root node cannot be selected
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.) #Sample node to be mutated

    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, mutate_idx))

    new_operator = jr.choice(sample_key, a=operator_indices, shape=(), p=operator_probabilities) #Sample new operator
    new_slots = slots[new_operator]

    subtree = sample_tree(subtree_key, 2, variable_array) #Sample new subtree
    subtree_size = jnp.sum(subtree[:,0]!=0)
    tree_size = mutate_idx - end_idx

    second_branch = jr.bernoulli(side_key) #Sample if the old subtree is the first or second operand

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child = jnp.where(tree_indices > mutate_idx, tree, child) #Insert nodes before the mutation index in the new tree
    child = jnp.where(tree_indices < end_idx - (new_slots - 1) * subtree_size, jnp.roll(tree, -(new_slots - 1) * subtree_size - 1, axis=0), child) #Insert nodes after the subtree in the new tree
    child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] - (new_slots - 1) * subtree_size - 1, child[:,1:3])) #Update index references

    rolled_tree = jnp.roll(tree, - (new_slots - 1) * second_branch * subtree_size - 1, axis=0) #Align position of the old subtree with the new tree
    rolled_tree = rolled_tree.at[:,1:3].set(jnp.where(rolled_tree[:,1:3] > -1, rolled_tree[:,1:3] - 1 - (new_slots - 1) * second_branch*subtree_size, rolled_tree[:,1:3])) #Update index references in old subtree

    rolled_subtree = jnp.roll(subtree, mutate_idx - max_nodes - (1-second_branch) * tree_size, axis=0) #Align position of the new subtree with the new tree
    rolled_subtree = rolled_subtree.at[:,1:3].set(jnp.where(rolled_subtree[:,1:3] > -1, rolled_subtree[:,1:3] - (max_nodes - mutate_idx) - (1-second_branch)*tree_size, rolled_subtree[:,1:3])) #Update index references in new subtree

    lower_tree = jax.lax.select(second_branch, jnp.where(tree_indices <= mutate_idx - subtree_size - 1, rolled_tree, rolled_subtree), 
                            jnp.where(tree_indices <= end_idx - 1, rolled_subtree, rolled_tree)) #Place first and second subtree
    
    child_2_branches = jnp.where((tree_indices <= mutate_idx - 1) & (tree_indices > mutate_idx - subtree_size - tree_size - 1), lower_tree, child) #Insert subtrees in new tree

    child_1_branch = jnp.where((tree_indices <= mutate_idx - 1) & (tree_indices >= mutate_idx - tree_size), rolled_tree, child) #Insert old subtree in new tree
    
    child = jax.lax.select(new_slots==2, child_2_branches, child_1_branch) #Select tree with one or two operands
    child = child.at[mutate_idx, 0].set(new_operator)
    child = child.at[mutate_idx, 1].set(mutate_idx - 1)
    child = child.at[mutate_idx, 2].set(jax.lax.select(new_slots==2, mutate_idx - jax.lax.select(second_branch, subtree_size, tree_size) - 1, -1))

    return child

def replace_tree(tree: Array, 
                 key: PRNGKey, 
                 variable_array: Array, 
                 args: Tuple) -> Array:
    """
    Samples a new tree
    
    :param tree
    :param key
    :param variable_array: The valid variables for this tree
    :param args: Miscellaneous parameters required for mutation

    Returns: Sampled tree
    """
    (sample_tree, max_nodes, max_init_depth, variable_indices, operator_indices, operator_probabilities, slots, coefficient_sd) = args
    return sample_tree(key, max_init_depth, variable_array)

def mutate_tree(tree: Array, 
                key: PRNGKey, 
                mutate_function: int, 
                variable_array: Array, 
                partial_mutate_functions: list[Callable]) -> Array:
    """
    Apply mutation to tree
    
    :param tree
    :param key
    :param mutate_function: Indicates which mutation should be applied
    :param variable_array: The valid variables for this tree
    :partial_mutate_functions: Mutation functions that can be applied

    Returns: Mutated tree
    """
    return jax.lax.switch(mutate_function, partial_mutate_functions, tree, key, variable_array)

def get_mutations(tree: Array, 
                  key: PRNGKey) -> int:
    """
    Sample mutation function to apply to the tree
    
    :param tree
    :param key

    Returns: Index of mutation function
    """
        
    mutation_probs = jnp.ones(len(MUTATE_FUNCTIONS))
    mutation_probs = jax.lax.select(jnp.sum(tree[:,0]==0) < 8, jnp.array([0., 1., 1., 1., 0., 0., 1.]), mutation_probs) #Tree is too big to add more nodes
    mutation_probs = jax.lax.select(jnp.sum(tree[:,0]!=0) <= 3, jnp.array([1., 1., 1., 0., 1., 0., 1.]), mutation_probs) #Tree does not have enough operators
    mutation_probs = jax.lax.select(jnp.sum(tree[:,0]!=0) == 1, jnp.array([1., 1., 0., 0., 1., 0., 1.]), mutation_probs) #Tree does not have operators
    
    return jr.choice(key, jnp.arange(len(MUTATE_FUNCTIONS)), p=mutation_probs)

#Define list with possible mutation functions
MUTATE_FUNCTIONS = [add_subtree, mutate_leaf, mutate_operator, delete_operator, prepend_operator, insert_operator, replace_tree]

def initialize_mutation_functions(args: Tuple) -> Callable:
    """
    Initializes the mutation functions with static arguments

    :param args: Miscellaneous parameters required for mutation

    Returns: Jittable mutation function
    """

    partial_mutate_functions = [partial(f, args=args) for f in MUTATE_FUNCTIONS] #Set args as static argument in mutation functions

    def mutate_trees(trees: Array, 
                     keys: PRNGKey, 
                     reproduction_probability: float, 
                     variable_array: Array) -> Array:
        """
        Mutates the trees in a candidate
        
        :param tree
        :param key
        :param reproduction_probability: Probability of a tree to be mutated
        :param variable_array: The valid variables for this tree

        Returns: Mutated candidate
        """

        #Sample the trees that will be mutated. At least one tree will be mutated
        _, mutate_indices, _ = jax.lax.while_loop(lambda carry: jnp.sum(carry[1])==0, sample_indices, (keys[0], jnp.zeros(trees.shape[0]), reproduction_probability))
        mutate_functions = jax.vmap(get_mutations)(trees, keys)

        mutated_trees = jax.vmap(mutate_tree, in_axes=[0,0,0,0,None])(trees, keys, mutate_functions, variable_array, partial_mutate_functions)

        #Only keep the new trees of the mutation indices
        return jnp.where(mutate_indices[:,None,None] * jnp.ones_like(trees), mutated_trees, trees)
    
    return mutate_trees