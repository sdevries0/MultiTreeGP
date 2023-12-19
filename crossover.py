import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from jax import Array
import equinox as eqx

from networkTrees import NetworkTrees
import miscellaneous

def tree_intersection(tree_a: list, tree_b: list, path: list = [], interior_nodes: list = [], boundary_nodes: list = []):
    "Determines the intersecting nodes of a pair of trees. Specifies interior nodes with same arity and boundary nodes with different arity"
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

def get_subtree(tree: list, key: PRNGKey):
    "Return the subtree from a random node onwards"
    leaves = jax.tree_util.tree_leaves(tree)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    depths_per_node = miscellaneous.depth_per_node(tree)
    distribution = jnp.array([0.9**depth for depth in depths_per_node]) #increase selection probability of operators
    distribution = distribution.at[0].set(0.1) #lower probability of root node
    index = jrandom.choice(key, jnp.arange(len(leaves)),p=distribution)

    path = flat_tree_and_path[index][0][:-1]

    subtree = miscellaneous.key_loc(tree, path)

    return path, subtree

def standard_cross_over(trees_a: NetworkTrees, trees_b: NetworkTrees, reproduction_probabilities: Array, state_size: int, key: PRNGKey):
    "Performs standard cross-over on a pair of trees, returning two new trees. A cross-over point is selected in each pair of trees, interchanging the subtrees below this point"
    key, new_key = jrandom.split(key)
    crossover_bool = jrandom.bernoulli(new_key, p = reproduction_probabilities, shape=(state_size+1,))
    while jnp.sum(crossover_bool)==0: #Make sure that at least one tree is mutated
        key, new_key = jrandom.split(key)
        #Sample which pair of trees are selected for cross-over. Multiple pairs possible
        crossover_bool = jrandom.bernoulli(new_key, p = reproduction_probabilities, shape=(state_size+1,))
    for i in range(state_size):
        if crossover_bool[i]:
            #Apply cross-over to a pair of trees
            key, key_a, key_b = jrandom.split(key, 3)
            path_a, subtree_a = get_subtree(trees_a()[i], key_a)
            path_b, subtree_b = get_subtree(trees_b()[i], key_b)

            trees_a = eqx.tree_at(lambda t: miscellaneous.key_loc(t()[i], path_a), trees_a, subtree_b)
            trees_b = eqx.tree_at(lambda t: miscellaneous.key_loc(t()[i], path_b), trees_b, subtree_a)

    #Apply cross-over to readout trees
    if crossover_bool[-1]:
        key, key_a, key_b = jrandom.split(key, 3)
        path_a, subtree_a = get_subtree(trees_a.readout_tree, key_a)
        path_b, subtree_b = get_subtree(trees_b.readout_tree, key_b)

        trees_a = eqx.tree_at(lambda t: miscellaneous.key_loc(t.readout_tree, path_a), trees_a, subtree_b)
        trees_b = eqx.tree_at(lambda t: miscellaneous.key_loc(t.readout_tree, path_b), trees_b, subtree_a)

    return trees_a, trees_b

def tree_cross_over(trees_a: NetworkTrees, trees_b: NetworkTrees, reproduction_probabilities: Array, state_size: int, key: PRNGKey):
    "Applies cross-over on tree level, interchanging only full trees"
    new_trees_a = trees_a
    new_trees_b = trees_b

    key, new_key = jrandom.split(key)
    mutate_bool = jrandom.bernoulli(new_key, p = reproduction_probabilities, shape=(state_size+1,))
    while jnp.sum(mutate_bool)==0: #Make sure that at least one tree is mutated
        key, new_key = jrandom.split(key)
        #Sample which pair of trees are selected for cross-over. Multiple pairs possible
        mutate_bool = jrandom.bernoulli(new_key, p = reproduction_probabilities, shape=(state_size+1,))
    for i in range(state_size):
        if mutate_bool[i]:
            #Apply cross-over to a pair of trees
            new_trees_a = eqx.tree_at(lambda t: t()[i], new_trees_a, trees_b()[i])
            new_trees_b = eqx.tree_at(lambda t: t()[i], new_trees_b, trees_a()[i])
    if mutate_bool[-1]: #Apply cross-over to readout
        new_trees_a = eqx.tree_at(lambda t: t.readout_tree, new_trees_a, trees_b.readout_tree)
        new_trees_b = eqx.tree_at(lambda t: t.readout_tree, new_trees_b, trees_a.readout_tree)
    return new_trees_a, new_trees_b

def uniform_cross_over(trees_a: NetworkTrees, trees_b: NetworkTrees, reproduction_probabilities, state_size, key: PRNGKey):
    "Performs uniform cross-over on a pair of trees, returning two new trees. Each overlapping node is switched with 50% chance and children of boundary nodes are switched as well."
    key, new_key = jrandom.split(key)
    mutate_bool = jrandom.bernoulli(new_key, p = reproduction_probabilities, shape=(state_size+1,))
    while jnp.sum(mutate_bool)==0: #Make sure that at least one tree is mutated
        key, new_key = jrandom.split(key)
        #Sample which pair of trees are selected for cross-over. Multiple pairs possible
        mutate_bool = jrandom.bernoulli(new_key, p = reproduction_probabilities, shape=(state_size,))

    for i in range(state_size): #Get intersection of the trees
        if mutate_bool[i]:
            #Apply cross-over to a pair of trees
            interior_nodes, boundary_nodes = tree_intersection(trees_a()[i], trees_b()[i], path = [], interior_nodes = [], boundary_nodes = [])
            new_trees_a = trees_a
            new_trees_b = trees_b

            for node in interior_nodes: #Randomly switch two nodes of interior intersecting nodes
                key, new_key = jrandom.split(key)
                if jrandom.uniform(new_key) > 0.5:
                    new_trees_a = eqx.tree_at(lambda t: miscellaneous.index_loc(t()[i], node), new_trees_a, miscellaneous.index_loc(trees_b()[i], node))
                    new_trees_b = eqx.tree_at(lambda t: miscellaneous.index_loc(t()[i], node), new_trees_b, miscellaneous.index_loc(trees_a()[i], node))

            for node in boundary_nodes: #Randomly switch two nodes and their children of boundary intersecting nodes
                key, new_key = jrandom.split(key)
                if jrandom.uniform(new_key) > 0.5:
                    new_trees_a = eqx.tree_at(lambda t: miscellaneous.index_loc(t()[i], node[:-1]), new_trees_a, miscellaneous.index_loc(trees_b()[i], node[:-1]))
                    new_trees_b = eqx.tree_at(lambda t: miscellaneous.index_loc(t()[i], node[:-1]), new_trees_b, miscellaneous.index_loc(trees_a()[i], node[:-1]))
            
            trees_a = new_trees_a
            trees_b = new_trees_b

    #Apply cross-over to readout
    if mutate_bool[-1]:
        interior_nodes, boundary_nodes = tree_intersection(trees_a.readout_tree, trees_b.readout_tree, path = [], interior_nodes = [], boundary_nodes = []) 
        new_trees_a = trees_a
        new_trees_b = trees_b

        #Randomly switch two nodes of interior intersecting nodes
        for node in interior_nodes:
            key, new_key = jrandom.split(key)
            if jrandom.uniform(new_key) > 0.5:
                new_trees_a = eqx.tree_at(lambda t: miscellaneous.index_loc(t.readout_tree, node), new_trees_a, miscellaneous.index_loc(trees_b.readout_tree, node))
                new_trees_b = eqx.tree_at(lambda t: miscellaneous.index_loc(t.readout_tree, node), new_trees_b, miscellaneous.index_loc(trees_a.readout_tree, node))
        #Randomly switch two nodes and their children of boundary intersecting nodes
        for node in boundary_nodes:
            key, new_key = jrandom.split(key)
            if jrandom.uniform(new_key) > 0.5:
                new_trees_a = eqx.tree_at(lambda t: miscellaneous.index_loc(t.readout_tree, node[:-1]), new_trees_a, miscellaneous.index_loc(trees_b.readout_tree, node[:-1]))
                new_trees_b = eqx.tree_at(lambda t: miscellaneous.index_loc(t.readout_tree, node[:-1]), new_trees_b, miscellaneous.index_loc(trees_a.readout_tree, node[:-1]))
        
        trees_a = new_trees_a
        trees_b = new_trees_b

    return trees_a, trees_b