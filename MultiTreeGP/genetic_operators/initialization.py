import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
from functools import partial

def sample_FS_node(i, carry):
    key, matrix, open_slots, max_depth, max_nodes, var_prob, args = carry
    leaf_indices, func_indices, func_probabilities, slots, leaf_sd, map_b_to_d = args
    float_key, leaf_key, variable_key, node_key, func_key = jr.split(key, 5)
    _i = map_b_to_d[i].astype(int)

    depth = (jnp.log(i+1)/jnp.log(2)).astype(int)
    float = jr.normal(float_key)*leaf_sd
    leaf = jax.lax.select(jr.uniform(leaf_key)<0.5, 1, jr.choice(variable_key, leaf_indices, shape=(), p=var_prob))
    index = jax.lax.select((open_slots < max_nodes - i - 1) & (depth<max_depth), 
                            jax.lax.select(jr.uniform(node_key)<(0.7**depth), 
                                            jr.choice(func_key, a=func_indices, shape=(), p=func_probabilities), 
                                            leaf), 
                            leaf)
    index = jax.lax.select(open_slots == 0, 0, index)
    index = jax.lax.select(i>0, jax.lax.select((slots[jnp.maximum(matrix[map_b_to_d[(i + (i%2) - 2)//2].astype(int), 0], 0).astype(int)] + i%2) > 1, index, 0), index)

    matrix = jax.lax.select(slots[index] > 0, matrix.at[_i, 1].set(map_b_to_d[2*i+1]), matrix.at[_i, 1].set(-1))
    matrix = jax.lax.select(slots[index] > 1, matrix.at[_i, 2].set(map_b_to_d[2*i+2]), matrix.at[_i, 2].set(-1))

    matrix = jax.lax.select(index == 1, matrix.at[_i,3].set(float), matrix)
    matrix = matrix.at[_i, 0].set(index)

    open_slots = jax.lax.select(index == 0, open_slots, jnp.maximum(0, open_slots + slots[index] - 1))

    return (jr.fold_in(key, i), matrix, open_slots, max_depth, max_nodes, var_prob, args)

def prune_row(i, carry, old_matrix):
    matrix, counter, max_depth = carry

    _i = 2**max_depth - i - 2

    row = old_matrix[_i]

    matrix = jax.lax.select(row[0] != 0, matrix.at[counter].set(row), matrix.at[:,1:3].set(jnp.where(matrix[:,1:3] > _i, matrix[:,1:3]-1, matrix[:,1:3])))
    counter = jax.lax.select(row[0] != 0, counter - 1, counter)

    return (matrix, counter, max_depth)
    
def prune_tree(matrix, max_depth, max_nodes):
    matrix, counter, _ = jax.lax.fori_loop(0, 2**max_depth-1, partial(prune_row, old_matrix=matrix), (jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (max_nodes, 1)), max_nodes-1, max_depth))
    matrix = matrix.at[:,1:3].set(jnp.where(matrix[:,1:3]>-1, matrix[:,1:3] + counter + 1, matrix[:,1:3]))
    return matrix

def sample_tree(key, depth, var_prob, max_depth, max_nodes, args):
    tree = jax.lax.fori_loop(0, 2**max_depth-1, sample_FS_node, (key, jnp.zeros((2**max_depth-1, 4)), 1, depth, max_nodes, var_prob, args))[1]
    return prune_tree(tree, max_depth, max_nodes)

def sample_trees(keys, depth, max_depth, max_nodes, variable_probabilities, args):
    return jax.vmap(sample_tree, in_axes=[0, None, 0, None, None, None])(keys, depth, variable_probabilities, max_depth, max_nodes, args)

def sample_population(key, population_size, num_trees, max_init_depth, max_depth, max_nodes, variable_probabilities, args):
    return jax.vmap(sample_trees, in_axes=[0, None, None, None, None, None])(jr.split(key, (population_size, num_trees)), max_init_depth, max_depth, max_nodes, variable_probabilities, args)
    