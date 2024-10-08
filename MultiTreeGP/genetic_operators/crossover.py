import jax
import jax.numpy as jnp
import jax.random as jr

def sample_indices(carry):
    _key, prev, reproduction_probability = carry
    indices = jr.bernoulli(_key, p=reproduction_probability, shape=prev.shape)*1.0
    return (jr.split(_key, 1)[0], indices, reproduction_probability)

def find_end_idx(carry):
    tree, openslots, counter = carry
    _, idx1, idx2, _ = tree[counter]
    openslots -= 1
    openslots = jax.lax.select(idx1 < 0, openslots, openslots+1)
    openslots = jax.lax.select(idx2 < 0, openslots, openslots+1)
    counter -= 1
    return (tree, openslots, counter)

def check_equal_subtrees(i, carry):
    tree1, tree2, equal = carry

    same_leaf = (tree1[i,3]==tree2[i,3]) & (tree1[i,0]==1)
    equal = jax.lax.select(((tree1[i,0]==tree2[i,0]) & (tree1[i,0]>1)) | same_leaf, equal, False)

    return tree1, tree2, equal

def check_invalid_cx_points(carry):
    tree1, tree2, _, node_idx1, node_idx2, _, _ = carry

    _, _, end_idx1 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree1, 1, node_idx1))
    _, _, end_idx2 = jax.lax.while_loop(lambda carry: carry[1]>0, find_end_idx, (tree2, 1, node_idx2))

    subtree_size1 = node_idx1 - end_idx1
    subtree_size2 = node_idx2 - end_idx2

    empty_nodes1 = jnp.sum(tree1[:,0]==0)
    empty_nodes2 = jnp.sum(tree2[:,0]==0)

    equal_subtrees = jax.lax.select((subtree_size1==subtree_size2) & ((jnp.sum(tree1[:,0]!=0) > 1) | (jnp.sum(tree2[:,0]!=0) > 1)), 
                                    jax.lax.fori_loop(0, subtree_size1, check_equal_subtrees, (jnp.roll(tree1, -node_idx1 + subtree_size1 - 1, axis=0), jnp.roll(tree2, -node_idx2 + subtree_size2 - 1, axis=0), True))[2], 
                                    False)

    return (empty_nodes1 < subtree_size2 - subtree_size1) | (empty_nodes2 < subtree_size1 - subtree_size2) | equal_subtrees

def sample_cx_points(carry):
    tree1, tree2, keys, _, _, node_ids, func_indices = carry
    key1, key2 = keys

    cx_prob1 = jnp.isin(tree1[:,0], func_indices)
    cx_prob1 = jnp.where(tree1[:,0]==0, cx_prob1, cx_prob1+1)
    node_idx1 = jr.choice(key1, node_ids, p = cx_prob1*1.)

    cx_prob2 = jnp.isin(tree2[:,0], func_indices)
    cx_prob2 = jnp.where(tree2[:,0]==0, cx_prob2, cx_prob2+1)
    node_idx2 = jr.choice(key2, node_ids, p = cx_prob2*1.)

    return (tree1, tree2, jr.split(key1), node_idx1, node_idx2, node_ids, func_indices)

def crossover(tree1, tree2, keys, max_nodes, func_indices):
    node_ids = jnp.arange(max_nodes)
    #Define indices of the nodes
    tree_indices = jnp.tile(node_ids[:,None], reps=(1,4))
    key1, key2 = keys

    #Define last node in tree
    last_node_idx1 = jnp.sum(tree1[:,0]==0)
    last_node_idx2 = jnp.sum(tree2[:,0]==0)

    #Randomly select nodes for crossover
    _, _, _, node_idx1, node_idx2, _, _ = sample_cx_points((tree1, tree2, jr.split(key1), 0, 0, node_ids, func_indices))

    #Reselect until valid crossover points have been found
    _, _, _, node_idx1, node_idx2, _, _ = jax.lax.while_loop(check_invalid_cx_points, sample_cx_points, (tree1, tree2, jr.split(key2), node_idx1, node_idx2, node_ids, func_indices))

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

def crossover_trees(parent1, parent2, keys, reproduction_probability, func_indices, max_nodes):
    _, cx_indices, _ = jax.lax.while_loop(lambda carry: jnp.sum(carry[1])==0, sample_indices, (keys[0, 0], jnp.zeros(parent1.shape[0]), reproduction_probability))
    offspring1, offspring2 = jax.vmap(crossover, in_axes=[0,0,0,None,None])(parent1, parent2, keys, max_nodes, func_indices)
    child1 = jnp.where(cx_indices[:,None,None] * jnp.ones_like(parent1), offspring1, parent1)
    child2 = jnp.where(cx_indices[:,None,None] * jnp.ones_like(parent2), offspring2, parent2)
    return child1, child2, jnp.zeros((2, 2), dtype=int)