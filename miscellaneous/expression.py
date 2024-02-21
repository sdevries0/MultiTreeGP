from typing import Any
import jax
import jax.numpy as jnp

class Node:
    def __init__(self, f, string, arity):
        self.f = f
        self.string = string
        self.arity = arity

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

    def __init__(self, obs_size = 0, state_size = 0, control_size = 0, target_size = 0, include_unary = False, condition = None, operators_prob = None, leaf_prob = None):
        plus = Node(lambda x, y: x + y, "+", 2)
        minus = Node(lambda x, y: x - y, "-", 2)
        multiplication = Node(lambda x, y: x * y, "*", 2)
        division = Node(lambda x, y: x / y, "/", 2)
        power = Node(lambda x, y: x ** y, "**", 2)
        self.operators = [plus, minus, multiplication, division, power]

        if include_unary:
            sine = Node(lambda x: jnp.sin(x), "sin", 1)
            cosine = Node(lambda x: jnp.cos(x), "cos", 1)
            tanh = Node(lambda x: jnp.tanh(x), "tanh", 1)
            exp = Node(lambda x: jnp.exp(x), "exp", 1)
            squareroot = Node(lambda x: jnp.sqrt(x), "√", 1)
            self.operators.extend([sine, cosine])

        if operators_prob is None:
            self.operators_prob = jnp.ones(len(self.operators))
        else:
            assert len(operators_prob) == len(self.operators), "Length of probabilities does not match the number of operators"
            self.operators_prob = operators_prob
            

        self.leaf_nodes = []
        if obs_size > 0:
            self.leaf_nodes.extend([Node(lambda args: args["y"][i], "y" + str(i), 0) for i in range(obs_size)])
        if state_size > 0:
            self.state_variables = [Node(lambda args: args["a"][i], "a" + str(i), 0) for i in range(state_size)]
            self.leaf_nodes.extend(self.state_variables)
        if control_size > 0:
            self.leaf_nodes.extend([Node(lambda args: args["u"][i], "u" + str(i), 0) for i in range(control_size)])
        if target_size > 0:
            self.leaf_nodes.extend([Node(lambda args: args["tar"][i], "tar" + str(i), 0) for i in range(target_size)])

        if condition:
            self.condition = lambda tree: condition(self, tree)
        else:
            self.condition = lambda _: False
