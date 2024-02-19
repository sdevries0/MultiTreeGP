from typing import Any
import jax
import jax.numpy as jnp

class Node:
    def __init__(self, f, string):
        self.f = f
        self.string = string

    def __str__(self) -> str:
        return self.string
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args)
    
    def __eq__(self, _other: object) -> bool:
        if isinstance(_other,jax.numpy.ndarray):
            return False
        return self.string == _other.string

class Expression:
    """
        Container class for the expression used in parse trees
        Attributes:
            binary_operators_map (dict[str, lambda function]): A dictionary that maps strings to the corresponding functionality for binary operators
            unary_operators_map (dict[str, lambda function]): A dictionary that maps strings to the corresponding functionality for unary operators
            variables (list[string]): Variables that represent the observations of the environment
            state_variables (list[string]): Variables that represent the hidden neurons in the agent
            control_variables (list[string]): Variables that represent control signals
    """

    def __init__(self, obs_size = 0, state_size = 0, control_size = 0, target_size = 0, condition = None):
        plus = Node(lambda x, y: x + y, "+")
        minus = Node(lambda x, y: x - y, "-")
        multiplication = Node(lambda x, y: x * y, "*")
        division = Node(lambda x, y: x / y, "/")
        power = Node(lambda x, y: x ** y, "**")
        self.binary_operators = [plus, minus, multiplication, division, power]

        sine = Node(lambda x: jnp.sin(x), "sin")
        cosine = Node(lambda x: jnp.cos(x), "cos")
        tanh = Node(lambda x: jnp.tanh(x), "tanh")
        exp = Node(lambda x: jnp.exp(x), "exp")
        squareroot = Node(lambda x: jnp.sqrt(x), "√")
        self.unary_operators= [sine, cosine]

        self.leaf_nodes = []
        if obs_size > 0:
            self.leaf_nodes.extend([Node(lambda args: args["y"][i], "y" + str(i)) for i in range(obs_size)])
        if state_size > 0:
            self.state_variables = [Node(lambda args: args["a"][i], "a" + str(i)) for i in range(state_size)]
            self.leaf_nodes.extend(self.state_variables)
        if control_size > 0:
            self.leaf_nodes.extend([Node(lambda args: args["u"][i], "u" + str(i)) for i in range(control_size)])
        if target_size > 0:
            self.leaf_nodes.extend([Node(lambda args: args["tar"][i], "tar" + str(i)) for i in range(target_size)])

        if condition:
            self.condition = lambda tree: condition(self, tree)
        else:
            self.condition = lambda _: False
