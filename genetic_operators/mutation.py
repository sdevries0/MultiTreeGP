import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
from jax.random import PRNGKey
import equinox as eqx
from miscellaneous.expression import Expression
import miscellaneous.helper_functions as helper_functions
import genetic_operators.initialization as initialization

def insert_operator(expressions: Expression, tree: list, key: PRNGKey):
    "Insert an operator at a random point in tree. Sample a new leaf if necessary to satisfy arity of the operator"
    nodes = jax.tree_util.tree_leaves(tree)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    operator_indices = jnp.ravel(jnp.argwhere(jnp.array([helper_functions.is_operator(node, expressions) for node in nodes])))

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, operator_indices)
    path = flat_tree_and_path[index][0][:-1]
    subtree = helper_functions.key_loc(tree, path)
    
    new_key1, new_key2, new_key3 = jrandom.split(key, 3)
    new_operator = expressions.operators[jrandom.choice(new_key1, a = jnp.arange(len(expressions.operators)), shape=(), p = expressions.operators_prob)]

    if new_operator.arity == 2:
        tree_position = jrandom.randint(new_key2, shape=(), minval=0, maxval=2)
        other_leaf = initialization.sample_leaf(new_key3, expressions)
        new_tree = [new_operator, subtree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, subtree]
    elif new_operator.arity == 1:
        new_tree = [new_operator, subtree]
        
    return eqx.tree_at(lambda t: helper_functions.key_loc(t, path), tree, new_tree)

def add_subtree(expressions: Expression, tree: list, key: PRNGKey):
    #Replace a leaf with a new subtree
    nodes = jax.tree_util.tree_leaves(tree)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([helper_functions.is_leaf(node, expressions) for node in nodes])))

    key, new_key1, new_key2 = jrandom.split(key, 3)
    index = jrandom.choice(new_key1, leaf_indices)
    path = flat_tree_and_path[index][0][:-1]
    return eqx.tree_at(lambda t: helper_functions.key_loc(t, path), tree, initialization.grow_node(new_key2, expressions, depth=3))

def mutate_operator(expressions: Expression, tree: list, key: PRNGKey):
    "Replace an operator with different operator of equal arity"
    nodes = jax.tree_util.tree_leaves(tree)
    operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([helper_functions.is_operator(node, expressions) for node in nodes])))
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
        new_tree = eqx.tree_at(lambda t: helper_functions.key_loc(t, path), tree, new_operator)
    else:
        un_copy = [node for node in expressions.operators if node.arity == 1]
        un_copy.remove(symbol)
        key, new_key = jrandom.split(key)
        new_operator = un_copy[jrandom.randint(new_key, shape=(), minval=0, maxval=len(un_copy))]
        new_tree = eqx.tree_at(lambda t: helper_functions.key_loc(t, path), tree, new_operator)
    return eqx.tree_at(lambda t: t, tree, new_tree)

def prepend_operator(expressions: Expression, tree: list, key: PRNGKey):
    "Add an operator to the top of the tree"
    key, new_key = jrandom.split(key)
    new_operator = expressions.operators[jrandom.choice(new_key, a = jnp.arange(len(expressions.operators)), shape=(), p = expressions.operators_prob)]
    if new_operator.arity == 1:
        new_tree = [new_operator, tree]
    else:
        new_key1, new_key2 = jrandom.split(key, 2)
        tree_position = jrandom.randint(new_key1, shape=(), minval=0, maxval=2)
        #Sample a leaf for the other child of the operator
        other_leaf = initialization.sample_leaf(new_key2, expressions)

        new_tree = [new_operator, tree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, tree]
    return eqx.tree_at(lambda t: t, tree, new_tree)

def mutate_leaf(expressions: Expression, tree: list, key: PRNGKey):
    "Change value of a leaf. Leaf can stay the same type of change to a different leaf type"
    nodes = jax.tree_util.tree_leaves(tree)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([helper_functions.is_leaf(node, expressions) for node in nodes])))

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, leaf_indices)
    new_leaf = [nodes[index]]
    while new_leaf == [nodes[index]]:
        key, new_key1, new_key2 = jrandom.split(key, 3)
        index = jrandom.choice(new_key1, leaf_indices)
        new_leaf = initialization.sample_leaf(new_key2, expressions, sd=3)

    path = flat_tree_and_path[index][0][:-1]
    return eqx.tree_at(lambda t: helper_functions.key_loc(t, path), tree, new_leaf)

def mutate_constant(tree: list, key: PRNGKey):
    "Change the value of a constant leaf. The value is sampled close to the old value"
    nodes = jax.tree_util.tree_leaves(tree)
    constant_indicies = jnp.ravel(jnp.argwhere(jnp.array([helper_functions.is_constant(node) for node in nodes])))

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, constant_indicies)
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
    value = flat_tree_and_path[index][1]
    path = flat_tree_and_path[index][0]
    #Sample with old value as mean
    key, new_key = jrandom.split(key)
    return eqx.tree_at(lambda t: helper_functions.key_loc(t, path), tree, value+jrandom.normal(new_key))

def delete_operator(expressions: Expression, tree: list, key: PRNGKey):
    "Replace an operator with a new leaf"
    key, new_key = jrandom.split(key)
    new_leaf = initialization.sample_leaf(new_key, expressions, sd=3)
    nodes = jax.tree_util.tree_leaves(tree)
    operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([helper_functions.is_operator(node, expressions) for node in nodes])))
    flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)

    key, new_key = jrandom.split(key)
    index = jrandom.choice(new_key, operator_indicies)
    path = flat_tree_and_path[index][0][:-1]

    return eqx.tree_at(lambda t: helper_functions.key_loc(t, path), tree, new_leaf)

def mutate_tree(mutation_probabilities: Array, expressions: Expression, tree: list, key: PRNGKey, max_init_depth: int):
    #Applies on of the mutation types to a tree
    _mutation_probabilities = mutation_probabilities.copy()

    if len(tree)==1: #Tree does not contain operators, so exclude mutations that require operators
        _mutation_probabilities["mutate_operator"] = 0
        _mutation_probabilities["delete_operator"] = 0
        _mutation_probabilities["insert_operator"] = 0
    if sum([helper_functions.is_constant(node) for node in jax.tree_util.tree_leaves(tree)]) == 0: #Tree does not contain constants, so exclude mutations that require constants
        _mutation_probabilities["mutate_constant"] = 0

    key, new_key1, new_key2 = jrandom.split(key, 3)
    types, probabilities = list(_mutation_probabilities.keys()), list(_mutation_probabilities.values())
    mutation_type = types[jrandom.choice(new_key1, len(types), shape=(), p=jnp.array(probabilities))] #Sample with mutation type will be applied to the tree
    if mutation_type=="mutate_operator":
        new_tree = mutate_operator(expressions, tree, new_key2)
    elif mutation_type=="delete_operator":
        new_tree = delete_operator(expressions, tree, new_key2)
    elif mutation_type=="prepend_operator":
        new_tree = prepend_operator(expressions, tree, new_key2)
    elif mutation_type=="insert_operator":
        new_tree = insert_operator(expressions, tree, new_key2)
    elif mutation_type=="mutate_constant":
        new_tree = mutate_constant(tree, new_key2)
    elif mutation_type=="mutate_leaf":
        new_tree = mutate_leaf(expressions, tree, new_key2)
    elif mutation_type=="sample_subtree":
        new_tree = initialization.grow_node(new_key2, expressions, depth=max_init_depth)
    elif mutation_type=="add_subtree":
        new_tree = add_subtree(expressions, tree, new_key2)        
    return new_tree

def mutate_trees(parent: list, layer_sizes, key: PRNGKey, reproduction_probability: float, mutation_probabilities: Array, expressions: Expression, max_init_depth: int):
    key, new_key = jrandom.split(key)
    mutate_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(jnp.sum(layer_sizes),))
    while jnp.sum(mutate_bool)==0: #Make sure that at least one tree is mutated
        key, new_key = jrandom.split(key)
        mutate_bool = jrandom.bernoulli(new_key, p = reproduction_probability, shape=(jnp.sum(layer_sizes),))

    child = parent
    for i in range(layer_sizes.shape[0]):
        for j in range(layer_sizes[i]):
            if i > 0:
                index = layer_sizes[i-1] + j
            else:
                index = j

            if mutate_bool[index]:
                key, new_key = jrandom.split(key)
                new_tree = mutate_tree(mutation_probabilities, expressions[i], parent()[i][j], key, max_init_depth)
                child = eqx.tree_at(lambda t: t()[i][j], child, new_tree)
    return child