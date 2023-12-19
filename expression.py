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

    def __init__(self, binary_operators_map, unary_operators_map, variables, state_variables, control_variables):
        self.binary_operators_map = binary_operators_map
        self.unary_operators_map = unary_operators_map
        self.variables = variables
        self.state_variables = state_variables
        self.control_variables = control_variables

        self.binary_operators = list(self.binary_operators_map.keys()) #Store a list with only the string notations
        self.unary_operators = list(self.unary_operators_map.keys()) #Store a list with only the string notations