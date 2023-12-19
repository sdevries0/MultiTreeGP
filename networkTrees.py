import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

class NetworkTrees:
    """
        Class that contains trees that represent the hidden state and readout of a model
        Attributes:
            hidden_state_trees (list[PyTree]): A list of trees for each of the neurons in the hidden state
            readout_tree (PyTree): A tree that outputs a control force
            fitness (float)
    """

    def __init__(self, hidden_state_trees: list, readout_tree: list):
        self.hidden_state_trees = hidden_state_trees
        self.readout_tree = readout_tree
        self.fitness = None
    
    def __call__(self):
        return self.hidden_state_trees
    
    def set_fitness(self, fitness: float):
        self.fitness = fitness

    def get_params(self, values_and_paths: list):
        "Returns all optimisable parameters ##NOT USED"
        params = []
        paths = []
        for path, value in values_and_paths:
            if isinstance(value, jax.numpy.ndarray):
                params.append(value)
                paths.append(path)
        return jnp.array(params), paths
    
#Register the class of trees as a pytree
register_pytree_node(NetworkTrees, lambda tree: ((tree.hidden_state_trees,tree.readout_tree), None),
    lambda _, args: NetworkTrees(*args))