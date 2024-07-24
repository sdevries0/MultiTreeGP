import jax
import jax.numpy as jnp
import equinox as eqx
import re
from jax.tree_util import register_pytree_node
from MultiTreeGP.genetic_operators.simplification import tree_to_sympy

class TreePolicy:
    """
        Class that contains trees that represent the hidden state and readout of a model
        Attributes:
            hidden_state_trees (list[PyTree]): A list of trees for each of the neurons in the hidden state
            readout_tree (PyTree): A tree that outputs a control force
            fitness (float)
    """

    def __init__(self, trees):
        self.trees = trees
        self.fitness = None
    
    def __call__(self):
        return self.trees
    
    def set_fitness(self, fitness: float):
        self.fitness = fitness

    def reset_fitness(self):
        self.fitness = None
        
    def evaluate_tree(self, tree, expressions):
        """expression tree evaluation with variables x as input"""
        if len(tree) > 1:
            f = tree[0]
            nodes = [self.evaluate_tree(branch, expressions) for branch in tree[1:]]
            return lambda args: f(*map(lambda n: n(args), nodes))
        else:
            leaf = tree[0]
            if isinstance(leaf,jnp.ndarray):
                return lambda args: leaf
            return leaf

    def tree_to_function(self, expressions):
        functions = []
        for i in range(len(self.trees)):
            layer = self.trees[i]
            layer_functions = []
            for tree in layer:
                layer_functions.append(self.evaluate_tree(tree, expressions[i]))

            def f(layer_functions):
                return lambda args: jnp.array([func(args) for func in layer_functions])
            
            functions.append(f(layer_functions))
        return functions

    def get_params(self):
        "Returns all optimisable parameters ##NOT USED"
        values_and_paths = jax.tree_util.tree_leaves_with_path(self.trees)
        params = []
        paths = []
        for path, value in values_and_paths:
            if isinstance(value, jax.numpy.ndarray):
                params.append(value)
                paths.append(path)
        return jnp.array(params), paths
    
    def set_params(self, params, paths):
        "Set new values of optimisable parameters"
        for value, path in zip(params, paths):
            self.trees = eqx.tree_at(lambda t: self.key_loc(t, path), self.trees, value)

    def format_numbers(self, text):
        # Regular expression to find numbers with decimals
        pattern = r'\d+\.\d+'

        def replace(match):
            # Convert matched number to float, format it, and return as string
            return "{:.{}f}".format(float(match.group()), 3)

        # Use re.sub() to replace matched numbers with formatted ones
        return re.sub(pattern, replace, text)
        
    def __str__(self) -> str:
        string_output = ""
        for i in range(len(self.trees)):
            string_output += "["
            layer = self.trees[i]
            for j in range(len(layer)):
                string_output += str(tree_to_sympy(layer[j]))
                if j < (len(layer) - 1):
                    string_output += ", "
            string_output += "]"
            if i < (len(self.trees) - 1):
                string_output += ", "
        return self.format_numbers(string_output)
    
    def key_loc(self, tree: list, path: list):
        "Returns subtree at location specified by a path"
        new_tree = tree
        for k in path:
            new_tree = new_tree[k.idx]
        return new_tree

#Register the class of trees as a pytree
register_pytree_node(TreePolicy, lambda tree: ((tree()), None),
    lambda _, args: TreePolicy(args))