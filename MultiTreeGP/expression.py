from typing import Any
import jax
import jax.numpy as jnp

class OperatorNode:
    def __init__(self, f, string, arity, in_front=True):
        self.f = f
        self.string = string
        self.arity = arity
        self.in_front = in_front

    def __str__(self) -> str:
        return self.string
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args)
    
    def __eq__(self, _other: object) -> bool:
        if isinstance(_other,jax.numpy.ndarray):
            return False
        return self.string == _other.string

class LeafNode:
    def __init__(self, var, index):
        self.string = var + str(index)
        self.arity = 0

    def __str__(self) -> str:
        return self.string
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.f(*args)
    
    def __eq__(self, _other: object) -> bool:
        if isinstance(_other,jax.numpy.ndarray):
            return False
        return self.string == _other.string

OPERATORS = {}
OPERATORS["+"] = OperatorNode(lambda x, y: x + y, "+", 2)
OPERATORS["-"] = OperatorNode(lambda x, y: x - y, "-", 2)
OPERATORS["*"] = OperatorNode(lambda x, y: x * y, "*", 2)
OPERATORS["/"] = OperatorNode(lambda x, y: x / y, "/", 2)
OPERATORS["power"] = OperatorNode(lambda x, y: x ** y, "**", 2)
OPERATORS["sin"] = OperatorNode(lambda x: jnp.sin(x), "sin", 1)
OPERATORS["cos"] = OperatorNode(lambda x: jnp.cos(x), "cos", 1)
OPERATORS["tanh"] = OperatorNode(lambda x: jnp.tanh(x), "tanh", 1)
OPERATORS["exp"] = OperatorNode(lambda x: jnp.exp(x), "exp", 1)
OPERATORS["log"] = OperatorNode(lambda x: jnp.log(x), "log", 1)
OPERATORS["squareroot"] = OperatorNode(lambda x: jnp.sqrt(x), "√", 1)
OPERATORS["square"] = OperatorNode(lambda x: x ** 2, "squared", 1, in_front=False)

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

    def __init__(self, variables, operator_types = None, operator_probs = None, condition = None):
        if operator_types == None:
            operator_types = ["+", "-", "*", "/", "power"]

        operators = []
        for op_string in operator_types:
            operators.append(OPERATORS[op_string])
        self.operators = operators

        if operator_probs is None:
            self.operators_prob = jnp.ones(len(self.operators))
        else:
            assert len(operator_probs) == len(self.operators), "Length of probabilities does not match the number of operators"
            self.operators_prob = operator_probs

        self.leaf_nodes = []
        for var_string, var_size in variables:
            self.leaf_nodes.extend([LeafNode(var_string, i) for i in range(var_size)])

        if condition:
            self.condition = lambda tree: condition(tree)
        else:
            self.condition = lambda _: True
