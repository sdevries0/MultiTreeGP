import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial

def find_end_idx(carry):
    tree, openslots, counter = carry
    _, idx1, idx2, _ = tree[counter]
    openslots -= 1
    openslots = jax.lax.select(idx1 < 0, openslots, openslots+1)
    openslots = jax.lax.select(idx2 < 0, openslots, openslots+1)
    counter -= 1
    return (tree, openslots, counter)

def sample_indices(carry):
    _key, prev, reproduction_probability = carry
    indices = jr.bernoulli(_key, p=reproduction_probability, shape=prev.shape)*1.0
    return (jr.split(_key, 1)[0], indices, reproduction_probability)

def sample_leaf_point(carry):
    tree, key, _, _, variable_probabilities, leaf_indices = carry
    key, select_key, sample_key, variable_key = jr.split(key, 4)
    node_ids = tree[:,0]
    is_leaf = (node_ids == 1) | jnp.isin(node_ids, leaf_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.)
    new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, 1, jr.choice(variable_key, leaf_indices, shape=(), p=variable_probabilities))

    return (tree, key, mutate_idx, new_leaf, variable_probabilities, leaf_indices)

def check_equal_leaves(carry):
    tree, key, mutate_idx, new_leaf, _, _ = carry

    return (tree[mutate_idx, 0] == new_leaf) & (new_leaf != 1)

def check_invalid_operator_point(carry):
    tree, _, mutate_idx, new_operator, slots, _, _ = carry
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, mutate_idx))

    subtree_size = mutate_idx - end_idx

    empty_nodes = jnp.sum(tree[:,0]==0)
    new_tree_size = jax.lax.select(slots[new_operator] == 2, 7, 8)

    return (tree[mutate_idx, 0] == new_operator) | (empty_nodes + subtree_size < new_tree_size)

def sample_operator_point(carry):
    tree, key, _, _, slots, func_indices, func_probabilities = carry
    key, select_key, sample_key = jr.split(key, 3)

    node_ids = tree[:,0]
    is_operator = jnp.isin(node_ids, func_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.)
    new_operator = jr.choice(sample_key, a=func_indices, shape=(), p=func_probabilities)

    return (tree, key, mutate_idx, new_operator, slots, func_indices, func_probabilities)

def add_subtree(tree, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    select_key, sample_key = jr.split(key, 2)
    node_ids = tree[:,0]
    is_leaf = (node_ids == 1) | jnp.isin(node_ids, leaf_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.)

    subtree = sample_tree(sample_key, 2, variable_probabilities)
    subtree_size = jnp.sum(subtree[:,0]!=0)

    remaining_size = mutate_idx - jnp.sum(tree[:,0]==0)

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child = jnp.where(tree_indices > mutate_idx, tree, child)

    rolled_tree = jnp.roll(tree, -subtree_size + 1, axis=0)
    child = jnp.where((tree_indices <= mutate_idx - subtree_size) & (tree_indices > mutate_idx - subtree_size - remaining_size), rolled_tree, child)
    child = child.at[:,1:3].set(jnp.where((child[:,1:3] < (mutate_idx)) & (child[:,1:3] > -1), child[:,1:3] - (subtree_size - 1), child[:,1:3]))

    subtree = jnp.roll(subtree, -(max_nodes - mutate_idx - 1), axis=0)
    subtree = subtree.at[:,1:3].set(jnp.where(subtree[:,1:3] > -1, subtree[:,1:3] + (mutate_idx - max_nodes + 1), -1))
    child = jnp.where((tree_indices <= mutate_idx) & (tree_indices > mutate_idx - subtree_size), subtree, child)

    return child

def mutate_leaf(tree, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    select_key, sample_key, float_key, variable_key = jr.split(key, 4)
    node_ids = tree[:,0]
    is_leaf = (node_ids == 1) | jnp.isin(node_ids, leaf_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.)
    new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, 1, jr.choice(variable_key, leaf_indices, shape=(), p=variable_probabilities))
    _, _, mutate_idx, new_leaf, _, _ = jax.lax.while_loop(check_equal_leaves, sample_leaf_point, (tree, jr.fold_in(key, 0), mutate_idx, new_leaf, variable_probabilities, leaf_indices))

    float = jr.normal(float_key)

    child = tree.at[mutate_idx, 0].set(new_leaf)
    child = jax.lax.select(new_leaf==1, child.at[mutate_idx, 3].set(float), child.at[mutate_idx, 3].set(0))

    return child

def replace_with_one_subtree(tree, mutate_idx, operator, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, mutate_idx))

    remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1

    subtree = sample_tree(key, 2, variable_probabilities)
    subtree_size = jnp.sum(subtree[:,0]!=0)

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))

    child = jnp.where(tree_indices >= mutate_idx, tree, child)

    rolled_tree = jnp.roll(tree, (mutate_idx - end_idx - subtree_size - 1), axis=0)
    child = jnp.where((tree_indices < mutate_idx - subtree_size) & (tree_indices >= mutate_idx - subtree_size - remaining_size), rolled_tree, child)

    child = child.at[mutate_idx, 0].set(operator)
    child = child.at[mutate_idx, 2].set(-1)

    child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] + (mutate_idx - end_idx - subtree_size - 1), child[:,1:3]))

    subtree = jnp.roll(subtree, -(max_nodes - mutate_idx), axis=0)
    subtree = subtree.at[:,1:3].set(jnp.where(subtree[:,1:3] > -1, subtree[:,1:3] + (mutate_idx - max_nodes), -1))
    child = jnp.where((tree_indices < mutate_idx) & (tree_indices > mutate_idx - subtree_size - 1), subtree, child)

    return child

def replace_with_two_subtrees(tree, mutate_idx, operator, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    key1, key2 = jr.split(key)
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, mutate_idx))

    remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1

    subtree1 = sample_tree(key1, 1, variable_probabilities)
    subtree1_size = jnp.sum(subtree1[:,0]!=0)
    subtree2 = sample_tree(key2, 1, variable_probabilities)
    subtree2_size = jnp.sum(subtree2[:,0]!=0)

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child = jnp.where(tree_indices >= mutate_idx, tree, child)

    rolled_tree = jnp.roll(tree, (mutate_idx - end_idx - subtree1_size - subtree2_size - 1), axis=0)
    child = jnp.where((tree_indices < mutate_idx - subtree1_size - subtree2_size) & (tree_indices >= mutate_idx - subtree1_size - subtree2_size - remaining_size), rolled_tree, child)

    child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] + (mutate_idx - end_idx - subtree1_size - subtree2_size - 1), child[:,1:3]))

    child = child.at[mutate_idx, 0].set(operator)
    child = child.at[mutate_idx, 1].set(mutate_idx - 1)
    child = child.at[mutate_idx, 2].set(mutate_idx - subtree1_size - 1)

    subtree1 = jnp.roll(subtree1, -(max_nodes - mutate_idx), axis=0)
    subtree1 = subtree1.at[:,1:3].set(jnp.where(subtree1[:,1:3] > -1, subtree1[:,1:3] + (mutate_idx - max_nodes), -1))
    child = jnp.where((tree_indices < mutate_idx) & (tree_indices > mutate_idx - subtree1_size - 1), subtree1, child)

    subtree2 = jnp.roll(subtree2, -(max_nodes - mutate_idx + subtree1_size), axis=0)
    subtree2 = subtree2.at[:,1:3].set(jnp.where(subtree2[:,1:3] > -1, subtree2[:,1:3] + (mutate_idx - subtree1_size - max_nodes), -1))
    child = jnp.where((tree_indices < mutate_idx - subtree1_size) & (tree_indices > mutate_idx - subtree1_size - subtree2_size - 1), subtree2, child)

    return child

def mutate_operator(tree, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    select_key, sample_key, subtree_key = jr.split(key, 3)
    node_ids = tree[:,0]
    is_operator = jnp.isin(node_ids, func_indices)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.)
    new_operator = jr.choice(sample_key, a=func_indices, shape=(), p=func_probabilities)

    _, _, mutate_idx, new_operator, _, _, _ = jax.lax.while_loop(check_invalid_operator_point, sample_operator_point, (tree, jr.fold_in(key, 0), mutate_idx, new_operator, slots, func_indices, func_probabilities))

    current_slots = slots[node_ids[mutate_idx].astype(int)]
    new_slots = slots[new_operator]

    child = jax.lax.select(current_slots==2, jax.lax.select(new_slots==2, tree.at[mutate_idx, 0].set(new_operator), replace_with_one_subtree(tree, mutate_idx, new_operator, subtree_key, variable_probabilities, args)), 
                                jax.lax.select(new_slots==2, replace_with_two_subtrees(tree, mutate_idx, new_operator, subtree_key, variable_probabilities, args), tree.at[mutate_idx, 0].set(new_operator)))

    return child

def delete_operator(tree, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    select_key, sample_key, float_key, variable_key = jr.split(key, 4)
    node_ids = tree[:,0]
    is_operator = jnp.isin(node_ids, func_indices)
    is_operator = is_operator.at[-1].set(False)
    delete_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.)
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, delete_idx))

    remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1

    float = jr.normal(float_key)
    new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, 1, jr.choice(variable_key, leaf_indices, shape=(), p=variable_probabilities))

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child = jnp.where(tree_indices > delete_idx, tree, child)

    rolled_tree = jnp.roll(tree, delete_idx - end_idx - 1, axis=0)
    child = jnp.where((tree_indices < delete_idx) & (tree_indices >= delete_idx - remaining_size), rolled_tree, child)
    child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (delete_idx - 1)) & (child[:,1:3] > -1), child[:,1:3] + (delete_idx - end_idx - 1), child[:,1:3]))

    child = child.at[delete_idx, 0].set(new_leaf)
    child = jax.lax.select(new_leaf==1, child.at[delete_idx, 3].set(float), child.at[delete_idx, 3].set(0))

    return child

def prepend_operator(tree, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    sample_key, subtree_key, side_key = jr.split(key, 3)
    new_operator = jr.choice(sample_key, a=func_indices, shape=(), p=func_probabilities)
    new_slots = slots[new_operator]
    subtree = sample_tree(subtree_key, 2, variable_probabilities)
    subtree_size = jnp.sum(subtree[:,0]!=0)
    tree_size = jnp.sum(tree[:,0]!=0)

    second_branch = jr.bernoulli(side_key)

    child = jnp.roll(tree, -1 - (new_slots - 1) * second_branch*subtree_size, axis=0)
    child = child.at[:,1:3].set(jnp.where(child[:,1:3] > -1, child[:,1:3] - 1 - (new_slots - 1) * second_branch*subtree_size, child[:,1:3]))

    rolled_subtree = jnp.roll(subtree, -1 - (1-second_branch) * tree_size, axis=0)
    rolled_subtree = rolled_subtree.at[:,1:3].set(jnp.where(rolled_subtree[:,1:3] > -1, rolled_subtree[:,1:3] - 1 - (1-second_branch)*tree_size, rolled_subtree[:,1:3]))

    child_2_branches = jax.lax.select(second_branch, jnp.where((tree_indices < max_nodes - 1) & (tree_indices >= max_nodes - subtree_size - 1), rolled_subtree, child), jnp.where((tree_indices < max_nodes - tree_size - 1) & (tree_indices >= max_nodes - tree_size - subtree_size - 1), rolled_subtree, child))

    child = jax.lax.select(new_slots==2, child_2_branches, child)
    child = child.at[-1, 0].set(new_operator)
    child = child.at[-1, 1].set(max_nodes - 2)
    child = child.at[-1, 2].set(jax.lax.select(new_slots==2, max_nodes - jax.lax.select(second_branch, subtree_size, tree_size) - 2, -1))

    return child

def insert_operator(tree, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    tree_indices = jnp.tile(jnp.arange(max_nodes)[:,None], reps=(1,4))
    select_key, sample_key, subtree_key, side_key = jr.split(key, 4)
    node_ids = tree[:,0]
    is_operator = jnp.isin(node_ids, func_indices)
    is_operator = is_operator.at[-1].set(False)
    mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.)
    _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree, 1, mutate_idx))

    new_operator = jr.choice(sample_key, a=func_indices, shape=(), p=func_probabilities)
    new_slots = slots[new_operator]
    subtree = sample_tree(subtree_key, 2, variable_probabilities)
    subtree_size = jnp.sum(subtree[:,0]!=0)
    tree_size = mutate_idx - end_idx

    second_branch = jr.bernoulli(side_key)

    child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1))
    child = jnp.where(tree_indices > mutate_idx, tree, child)
    child = jnp.where(tree_indices < end_idx - (new_slots - 1) * subtree_size, jnp.roll(tree, -(new_slots - 1) * subtree_size - 1, axis=0), child)
    child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] - (new_slots - 1) * subtree_size - 1, child[:,1:3]))

    rolled_tree = jnp.roll(tree, - (new_slots - 1) * second_branch * subtree_size - 1, axis=0)
    rolled_tree = rolled_tree.at[:,1:3].set(jnp.where(rolled_tree[:,1:3] > -1, rolled_tree[:,1:3] - 1 - (new_slots - 1) * second_branch*subtree_size, rolled_tree[:,1:3]))

    rolled_subtree = jnp.roll(subtree, mutate_idx - max_nodes - (1-second_branch) * tree_size, axis=0)
    rolled_subtree = rolled_subtree.at[:,1:3].set(jnp.where(rolled_subtree[:,1:3] > -1, rolled_subtree[:,1:3] - (max_nodes - mutate_idx) - (1-second_branch)*tree_size, rolled_subtree[:,1:3]))

    lower_tree = jax.lax.select(second_branch, jnp.where(tree_indices <= mutate_idx - subtree_size - 1, rolled_tree, rolled_subtree), 
                            jnp.where(tree_indices <= end_idx - 1, rolled_subtree, rolled_tree))
    
    child_2_branches = jnp.where((tree_indices <= mutate_idx - 1) & (tree_indices > mutate_idx - subtree_size - tree_size - 1), lower_tree, child)

    child_1_branch = jnp.where((tree_indices <= mutate_idx - 1) & (tree_indices >= mutate_idx - tree_size), rolled_tree, child)
    
    child = jax.lax.select(new_slots==2, child_2_branches, child_1_branch)
    child = child.at[mutate_idx, 0].set(new_operator)
    child = child.at[mutate_idx, 1].set(mutate_idx - 1)
    child = child.at[mutate_idx, 2].set(jax.lax.select(new_slots==2, mutate_idx - jax.lax.select(second_branch, subtree_size, tree_size) - 1, -1))

    return child

def replace_tree(tree, key, variable_probabilities, args):
    (sample_tree, max_nodes, max_init_depth, leaf_indices, func_indices, func_probabilities, slots) = args
    return sample_tree(key, max_init_depth, variable_probabilities)

def mutate_tree(tree, key, mutate_function, variable_probabilities, partial_mutate_functions):
    return jax.lax.switch(mutate_function, partial_mutate_functions, tree, key, variable_probabilities)

def get_mutations(tree, key):
    mutation_probs = jnp.ones(len(MUTATE_FUNCTIONS))
    mutation_probs = jax.lax.select(jnp.sum(tree[:,0]==0) < 8, jnp.array([0., 1., 1., 1., 0., 0., 1.]), mutation_probs)
    mutation_probs = jax.lax.select(jnp.sum(tree[:,0]!=0) <= 3, jnp.array([1., 1., 1., 0., 1., 0., 1.]), mutation_probs)
    mutation_probs = jax.lax.select(jnp.sum(tree[:,0]!=0) == 1, jnp.array([1., 1., 0., 0., 1., 0., 1.]), mutation_probs)
    
    return jr.choice(key, jnp.arange(len(MUTATE_FUNCTIONS)), p=mutation_probs)


MUTATE_FUNCTIONS = [add_subtree, mutate_leaf, mutate_operator, delete_operator, prepend_operator, insert_operator, replace_tree]

def initialize_mutation_function(args):
    partial_mutate_functions = [partial(f, args=args) for f in MUTATE_FUNCTIONS]
    def mutate_trees(trees, keys, reproduction_probability, variable_probabilities):
        _, mutate_indices, _ = jax.lax.while_loop(lambda carry: jnp.sum(carry[1])==0, sample_indices, (keys[0], jnp.zeros(trees.shape[0]), reproduction_probability))
        mutate_functions = jax.vmap(get_mutations)(trees, keys)

        mutated_trees = jax.vmap(mutate_tree, in_axes=[0,0,0,0,None])(trees, keys, mutate_functions, variable_probabilities, partial_mutate_functions)

        return jnp.where(mutate_indices[:,None,None] * jnp.ones_like(trees), mutated_trees, trees), mutate_functions
    
    return mutate_trees