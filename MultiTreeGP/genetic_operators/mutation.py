import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
from jax.random import PRNGKey
import equinox as eqx

from MultiTreeGP.expression import Expression
import MultiTreeGP.genetic_operators.initialization as initialization
from MultiTreeGP.networks.tree_policy import TreePolicy

def key_loc(tree: list, path: list) -> list:
    """Finds subtree at location specified by a path.

    :tree: Tree.
    :path: Path to node in the tree.
    :returns: Subtree.
    """
    new_tree = tree
    for k in path:
        new_tree = new_tree[k.idx]
    return new_tree

def insert_operator(expressions: Expression, tree: list, key: PRNGKey, leaf_sd: float) -> list:
    """Insert an operator at a random point in tree. Sample a new leaf if necessary to satisfy arity of the operator.

    :expressions: Expressions for each layer in a tree policy.
    :tree: Tree of operators and leaves.
    :key: Random key.
    :leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree.
    """
    nodes = jax.tree_util.tree_leaves(tree)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    operator_indices = jnp.ravel(jnp.argwhere(jnp.array([node in expressions.operators for node in nodes])))

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, operator_indices)
    path = flat_tree_and_path[index][0][:-1]
    subtree = key_loc(tree, path)
    
    new_key1, new_key2, new_key3 = jrandom.split(key, 3)
    new_operator = expressions.operators[jrandom.choice(new_key1, a = jnp.arange(len(expressions.operators)), shape=(), p = expressions.operators_prob)]

    if new_operator.arity == 2:
        tree_position = jrandom.randint(new_key2, shape=(), minval=0, maxval=2)
        other_leaf = initialization.sample_leaf(new_key3, expressions, leaf_sd)
        new_tree = [new_operator, subtree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, subtree]
    elif new_operator.arity == 1:
        new_tree = [new_operator, subtree]
        
    return eqx.tree_at(lambda t: key_loc(t, path), tree, new_tree)

def add_subtree(expressions: Expression, tree: list, key: PRNGKey, leaf_sd: float) -> list:
    """Replace a leaf with a new subtree.

    :expressions: Expressions for each layer in a tree policy.
    :tree: Tree of operators and leaves.
    :key: Random key.
    :leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree.
    """
    nodes = jax.tree_util.tree_leaves(tree)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([node not in expressions.operators for node in nodes])))

    key, new_key1, new_key2 = jrandom.split(key, 3)
    index = jrandom.choice(new_key1, leaf_indices)
    path = flat_tree_and_path[index][0][:-1]
    return eqx.tree_at(lambda t: key_loc(t, path), tree, initialization.grow_node(new_key2, expressions, depth=3, leaf_sd=leaf_sd))

def mutate_operator(expressions: Expression, tree: list, key: PRNGKey, leaf_sd: float) -> list:
    """Replace an operator with different operator of equal arity.

    :expressions: Expressions for each layer in a tree policy.
    :tree: Tree of operators and leaves.
    :key: Random key.
    :leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree.
    """
    nodes = jax.tree_util.tree_leaves(tree)
    operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([node in expressions.operators for node in nodes])))
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, operator_indicies)
    symbol = flat_tree_and_path[index][1]
    path = flat_tree_and_path[index][0]

    if symbol.arity == 2:
        bin_copy = [node for node in expressions.operators if node.arity == 2]
        bin_copy.remove(symbol)
        key, new_key = jrandom.split(key)
        new_operator = bin_copy[jrandom.randint(new_key, shape=(), minval=0, maxval=len(bin_copy))]
        new_tree = eqx.tree_at(lambda t: key_loc(t, path), tree, new_operator)
    else:
        un_copy = [node for node in expressions.operators if node.arity == 1]
        un_copy.remove(symbol)
        if len(un_copy)==0:
            new_key, new_key2, new_key3 = jrandom.split(key, 3)
            subtree = key_loc(tree, path[:-1])[1]

            bin_copy = [node for node in expressions.operators if node.arity == 2]
            new_operator = bin_copy[jrandom.randint(new_key, shape=(), minval=0, maxval=len(bin_copy))]

            tree_position = jrandom.randint(new_key2, shape=(), minval=0, maxval=2)
            other_leaf = initialization.sample_leaf(new_key3, expressions, sd=leaf_sd)
            new_subtree = [new_operator, subtree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, subtree]
            new_tree = eqx.tree_at(lambda t: key_loc(t, path[:-1]), tree, new_subtree)

        else:
            key, new_key = jrandom.split(key)
            new_operator = un_copy[jrandom.randint(new_key, shape=(), minval=0, maxval=len(un_copy))]
            new_tree = eqx.tree_at(lambda t: key_loc(t, path), tree, new_operator)

    return eqx.tree_at(lambda t: t, tree, new_tree)

def prepend_operator(expressions: Expression, tree: list, key: PRNGKey, leaf_sd: float) -> list:
    """Add an operator to the top of the tree.

    :expressions: Expressions for each layer in a tree policy.
    :tree: Tree of operators and leaves.
    :key: Random key.
    :leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree.
    """
    key, new_key = jrandom.split(key)
    new_operator = expressions.operators[jrandom.choice(new_key, a = jnp.arange(len(expressions.operators)), shape=(), p = expressions.operators_prob)]
    if new_operator.arity == 1:
        new_tree = [new_operator, tree]
    else:
        new_key1, new_key2 = jrandom.split(key, 2)
        tree_position = jrandom.randint(new_key1, shape=(), minval=0, maxval=2)
        #Sample a leaf for the other child of the operator
        other_leaf = initialization.sample_leaf(new_key2, expressions, leaf_sd)

        new_tree = [new_operator, tree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, tree]
    return eqx.tree_at(lambda t: t, tree, new_tree)

def mutate_leaf(expressions: Expression, tree: list, key: PRNGKey, leaf_sd: float) -> list:
    """Change value of a leaf. Leaf can stay the same type of change to a different leaf type.

    :param expressions: Expressions for each layer in a tree policy.
    :param tree: Tree of operators and leaves.
    :param key: Random key.
    :param leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree.
    """
    nodes = jax.tree_util.tree_leaves(tree)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([node not in expressions.operators for node in nodes])))

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, leaf_indices)
    new_leaf = [nodes[index]]
    while new_leaf == [nodes[index]]:
        key, new_key1, new_key2 = jrandom.split(key, 3)
        index = jrandom.choice(new_key1, leaf_indices)
        new_leaf = initialization.sample_leaf(new_key2, expressions, sd=leaf_sd)

    path = flat_tree_and_path[index][0][:-1]
    return eqx.tree_at(lambda t: key_loc(t, path), tree, new_leaf)

def mutate_constant(tree: list, key: PRNGKey, leaf_sd: float) -> list:
    """Change the value of a constant leaf. The value is sampled close to the old value.

    :param tree: Tree of operators and leaves.
    :param key: Random key.
    :param leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree.
    """
    nodes = jax.tree_util.tree_leaves(tree)
    constant_indicies = jnp.ravel(jnp.argwhere(jnp.array([isinstance(node, jax.numpy.ndarray) for node in nodes])))

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, constant_indicies)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    value = flat_tree_and_path[index][1]
    path = flat_tree_and_path[index][0]
    #Sample with old value as mean
    key, new_key = jrandom.split(key)
    return eqx.tree_at(lambda t: key_loc(t, path), tree, value+leaf_sd*jrandom.normal(new_key))

def delete_operator(expressions: Expression, tree: list, key: PRNGKey, leaf_sd: float) -> list:
    """Replace an operator with a new leaf.

    :param expressions: Expressions for each layer in a tree policy.
    :param tree: Tree of operators and leaves.
    :param key: Random key.
    :param leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree.
    """
    key, new_key = jrandom.split(key)
    new_leaf = initialization.sample_leaf(new_key, expressions, sd=leaf_sd)
    nodes = jax.tree_util.tree_leaves(tree)
    operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([node in expressions.operators for node in nodes])))
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, operator_indicies)
    path = flat_tree_and_path[index][0][:-1]

    return eqx.tree_at(lambda t: key_loc(t, path), tree, new_leaf)

def mutate_tree(mutation_probabilities: Array, expressions: Expression, tree: list, key: PRNGKey, max_init_depth: int, leaf_sd: float) -> list:
    """Applies on of the mutation functions to a tree.

    :param mutation_probabilities: Probabilities of the mutation funcitons. 
    :param expressions: Expressions for each layer in a tree policy.
    :param tree: Tree of operators and leaves.
    :param key: Random key.
    :param max_init_depth: Highest depth of a tree at initialization
    :param leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree.
    """
    _mutation_probabilities = mutation_probabilities.copy()

    if len(tree)==1: #Tree does not contain operators, so exclude mutations that require operators
        _mutation_probabilities["mutate_operator"] = 0
        _mutation_probabilities["delete_operator"] = 0
        _mutation_probabilities["insert_operator"] = 0
    if sum([isinstance(node, jax.numpy.ndarray) for node in jax.tree_util.tree_leaves(tree)]) == 0: #Tree does not contain constants, so exclude mutations that require constants
        _mutation_probabilities["mutate_constant"] = 0

    key, new_key1, new_key2 = jrandom.split(key, 3)
    types, probabilities = list(_mutation_probabilities.keys()), list(_mutation_probabilities.values())
    mutation_type = types[jrandom.choice(new_key1, len(types), shape=(), p=jnp.array(probabilities))] #Sample which mutation function will be applied to the tree
    if mutation_type=="mutate_operator":
        new_tree = mutate_operator(expressions, tree, new_key2, leaf_sd)
    elif mutation_type=="delete_operator":
        new_tree = delete_operator(expressions, tree, new_key2, leaf_sd)
    elif mutation_type=="prepend_operator":
        new_tree = prepend_operator(expressions, tree, new_key2, leaf_sd)
    elif mutation_type=="insert_operator":
        new_tree = insert_operator(expressions, tree, new_key2, leaf_sd)
    elif mutation_type=="mutate_constant":
        new_tree = mutate_constant(tree, new_key2, leaf_sd)
    elif mutation_type=="mutate_leaf":
        new_tree = mutate_leaf(expressions, tree, new_key2, leaf_sd)
    elif mutation_type=="sample_subtree":
        new_tree = initialization.grow_node(new_key2, expressions, depth=max_init_depth, leaf_sd=leaf_sd)
    elif mutation_type=="add_subtree":
        new_tree = add_subtree(expressions, tree, new_key2, leaf_sd)
    return new_tree

def mutate_trees(parent: TreePolicy, layer_sizes: Array, key: PRNGKey, reproduction_probability: float, mutation_probabilities: Array, expressions: list, max_init_depth: int, leaf_sd: float) -> TreePolicy:
    """Mutate trees in tree policy.

    :param parent: Tree policy to be mutated.
    :param layer_sizes: Size of each layer in a tree policy.
    :param key: Random key.
    :param reproduction_probability: Probability of a tree to be adapted in a tree policy.
    :param mutation_probabilities: Probabilities of the mutation functions.
    :param expressions: Expressions for each layer in a tree policy.
    :param max_init_depth: Highest depth of a tree during initialization.  
    :param leaf_sd: Standard deviation for sampling constants.
    :returns: Mutated tree policy.
    """
    key, new_key = jrandom.split(key)
    mutate_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(jnp.sum(layer_sizes),))
    while jnp.sum(mutate_bool)==0: #Make sure that at least one tree is mutated
        key, new_key = jrandom.split(key)
        mutate_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(jnp.sum(layer_sizes),))

    child = parent
    for i in range(layer_sizes.shape[0]):
        for j in range(layer_sizes[i]):
            if i > 0:
                index = jnp.sum(layer_sizes[:i]) + j
            else:
                index = j

            if mutate_bool[index]:
                key, new_key = jrandom.split(key)
                new_tree = mutate_tree(mutation_probabilities, expressions[i], parent()[i][j], key, max_init_depth, leaf_sd)
                child = eqx.tree_at(lambda t: t()[i][j], child, new_tree)
    return child