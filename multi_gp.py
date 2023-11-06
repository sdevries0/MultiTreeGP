import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jax.tree_util import register_pytree_node
import diffrax
import matplotlib.pyplot as plt
import time
from pathos.multiprocessing import ProcessingPool as Pool
import sympy
import copy
from typing import Callable, Sequence, Tuple
from jax.random import PRNGKey

class KeyGen:
    """
        Random key generator
        Attributes:
            key (PRNGKey)
    """
    def __init__(self, seed:int = 0):
        self._key = jrandom.PRNGKey(seed)

    def get(self) -> jrandom.PRNGKeyArray:
        self._key, new_key = jrandom.split(self._key)
        return new_key

class NetworkTrees:
    """
        Class that contains trees that represent the hidden state and readout of a model
        Attributes:
            hidden_state_trees (list[PyTree]): A list of trees for each of the neurons in the hidden state
            readout_tree (PyTree): A tree that outputs a control force
            fitness (float)
    """

    def __init__(self, hidden_state_trees:list, readout_tree:list):
        self.hidden_state_trees = hidden_state_trees
        self.readout_tree = readout_tree
        self.fitness = None
    
    def __call__(self):
        return self.hidden_state_trees
    
    def set_fitness(self, fitness:float):
        self.fitness = fitness

    def get_params(self, values_and_paths:list):
        "Returns all optimisable parameters ##NOT USED"
        params = []
        paths = []
        for path, value in values_and_paths:
            if isinstance(value, jax.numpy.ndarray):
                params.append(value)
                paths.append(path)
        return jnp.array(params), paths
    
    def _to_string(self, tree:list):
        "Transform tree to string"
        if len(tree)==3:
            left_tree = "(" + self._to_string(tree[1]) + ")" if len(tree[1]) == 3 else self._to_string(tree[1])
            right_tree = "(" + self._to_string(tree[2]) + ")" if len(tree[2]) == 3 else self._to_string(tree[2])
            return left_tree + tree[0] + right_tree
        elif len(tree)==2:
            return tree[0] + "(" + self._to_string(tree[1]) + ")"
        else:
            return str(tree[0])
    
    def __str__(self):
        print("to_string is used")
        return f"{[self._to_string(tree) for tree in self.hidden_state_trees]}, readout = {self._to_string(self.readout_tree)}"
    
#Register the class of trees as a pytree
register_pytree_node(NetworkTrees, lambda tree: ((tree.hidden_state_trees,tree.readout_tree), None),
    lambda _, args: NetworkTrees(*args))

class ODE_GP:
    def __init__(self, seed: int, state_size: int, tournament_size: int = 5, max_depth:int = 10, max_init_depth: int = 5, population_size: int = 100, num_populations: int = 1, migration_period: int = 5, migration_percentage: float = 0.1, migration_method: str = "ring", init_method: str = "ramped", similarity_threshold: float = 0.2, parsimony_punishment:int = 0.1, restart_iter_threshold: int = 8):
        """
        Genetic Programming algorithm to learn continuous-time controllers. Each candidate consists of a set of trees that represent the hidden state of a network and a tree that maps the hidden state to a control force. To generate a new population cross-over, mutation, reproduction and random sampling are applied to the current population.

        Attributes:
            state_size (int): Number of trees in the hidden state
            num_populations (int): Number of distributed subpopulations
            population_size (int): Number of candidates in each population
            max_depth (int): Max depth of a tree during evolution
            max_init_depth (int): Max depth of a tree at initialization
            init_method (string): Method used for initialiation of trees

            restart_iter_threshold (int): Threshold (defined in number of generations) that decides to restart a subpopulation when it does not improve in this period
            parsimony_punishment (float): Weight that punishes trees for their size.
            similarity_threshold (float): Threshold that decides when a pair of trees are similar
            tournament_size (int): Size of a tournament. Correlated with selection pressure
            tournament_indices (Array[int]): Indices of each solution in the tournament

            selection_pressures (Array[float]): Selection pressure used in tournament selection for each subpopulation
            tournament_probabilities (Array[float]): Probability to be selected for each candidate in a tournament   
            reproduction_probabilities (Array[float]): Probability of each of the reproduction mechanisms to be applied. Can be different for each subpopulation
            tree_mutate_probs (Array[float]): Probability to apply mutation on a tree for each subpopulation
            
            population_indices (Array[int]): Indices of each solution in the population
            migration_period (int): Number of generations after which the subpopulations migrate solutions to other subpopulations
            migration_size (int): Number of solutions that are migrated to other subpopulations
            migration_method (string): Method used for migrating solutions

            binary_operators_map (Map[Callable]): Maps binary function nodes to callable expressions
            unary_operators_map (Map[Callable]): Maps unary function nodes to callable expressions
            binary_operators (list[string]): Binary function nodes that can be included in trees
            unary_operators (list[string]): Unary function nodes that can be included in trees

            mutation_types (list[string]): Type of mutations that can be applied to trees
            mutation_probabilities (Array[float]): Probability to be selected for the mutation types
        """

        self.keyGen = KeyGen(seed)

        self.state_size = state_size
        self.num_populations = num_populations
        self.population_size = population_size
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.init_method = init_method

        self.restart_iter_threshold = restart_iter_threshold
        self.parsimony_punishment = parsimony_punishment
        self.similarity_threshold = similarity_threshold
        self.tournament_size = tournament_size
        self.tournament_indices = jnp.arange(self.tournament_size)

        # self.selection_pressures = jnp.linspace(0.6,1.0,self.num_populations)
        self.selection_pressures = jnp.linspace(0.9,0.9,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**self.tournament_indices for sp in self.selection_pressures])
        # self.reproduction_probabilities = jnp.vstack([jnp.linspace(0.2,0.7,self.num_populations),jnp.linspace(0.4,0.2,self.num_populations),
                                        #  jnp.linspace(0.5,0.0,self.num_populations),jnp.linspace(0.,0.1,self.num_populations)]).T
        self.reproduction_probabilities = jnp.vstack([jnp.linspace(0.45,0.45,self.num_populations),jnp.linspace(0.30,0.30,self.num_populations),
                                         jnp.linspace(0.2,0.2,self.num_populations),jnp.linspace(0.05,0.05,self.num_populations)]).T
        # self.tree_mutate_probs = jnp.linspace(0.5,0.2,self.num_populations)
        self.tree_mutate_probs = jnp.linspace(0.5, 0.5, self.num_populations)
        
        self.population_indices = jnp.arange(self.population_size)
        self.migration_period = migration_period
        self.migration_size = int(migration_percentage*self.population_size)
        self.migration_method = migration_method

        #Define operators
        self.binary_operators_map = {"+":lambda a,b:a+b, "-":lambda a,b:a-b, "*":lambda a,b:a*b, "/":lambda a,b:a/b, "**":lambda a,b:a**b}
        self.unary_operators_map = {"sin":lambda a:jnp.sin(a), "cos":lambda a:jnp.cos(a)}#,"exp":lambda a:jnp.clip(jnp.exp(a),a_max=100), "sqrt":lambda a:jnp.sqrt(jnp.abs(a))}
        self.binary_operators = list(self.binary_operators_map.keys())
        self.unary_operators = list(self.unary_operators_map.keys())

        #Define mutation types
        self.mutation_types = ["mutate_operator","delete_operator","insert_operator","mutate_constant","mutate_leaf","sample_subtree","prepend_operator","add_subtree","simplify_tree"]
        self.mutation_probabilities = jnp.ones(len(self.mutation_types))
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("mutate_operator")].set(1.0)
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("delete_operator")].set(0.5)
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("insert_operator")].set(1.0)
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("mutate_constant")].set(1.0)
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("mutate_leaf")].set(1.0)
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("sample_subtree")].set(1.0)
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("prepend_operator")].set(0.5)
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("add_subtree")].set(1.0)
        self.mutation_probabilities = self.mutation_probabilities.at[self.mutation_types.index("simplify_tree")].set(0.5)

    #Basic helper methods
    def _key_loc(self, tree: list, path: list):
        "Returns subtree at location specified by a path"
        new_tree = tree
        for k in path:
            new_tree = new_tree[k.idx]
        return new_tree
    
    def _index_loc(self, tree: list, path: list):
        "Returns subtree at location specified by indices"
        new_tree = tree
        for i in path:
            new_tree = new_tree[i]
        return new_tree
    
    def _is_operator(self, symbol):
        "Checks if node is an operator"
        return symbol in self.binary_operators or symbol in self.unary_operators
    
    def _is_leaf(self, symbol):
        "Checks if node is a leaf"
        return symbol not in self.binary_operators and symbol not in self.unary_operators
    
    def _is_constant(self, symbol):
        "Checks if node is a constant"
        return isinstance(symbol, jax.numpy.ndarray)

    def _to_string(self, tree: list):
        if len(tree)==3:
            if tree[1]==None or tree[2]==None:
                print(tree)
            left_tree = "(" + self._to_string(tree[1]) + ")" if len(tree[1]) == 3 else self._to_string(tree[1])
            right_tree = "(" + self._to_string(tree[2]) + ")" if len(tree[2]) == 3 else self._to_string(tree[2])
            return left_tree + tree[0] + right_tree
        elif len(tree)==2:
            return tree[0] + "(" + self._to_string(tree[1]) + ")"
        else:
            return str(tree[0])
        
    def _tree_depth(self, tree: list):
        "Returns highest depth of tree"
        flat_tree = jax.tree_util.tree_leaves_with_path(tree)
        return jnp.max(jnp.array([len(node[0]) for node in flat_tree]))
    
    def _depth_per_node(self, tree: list):
        "Returns depth of each node of the tree"
        flat_tree, = jax.tree_util.tree_leaves_with_path(tree)
        return jnp.array([len(node[0]) for node in flat_tree])
    
    def flatten(self, populations: list):
        #Flattens all subpopulations
        return [candidate for population in populations for candidate in population]

    def _best_solution(self, populations: list, fitnesses: Sequence):
        "Returns the best solution in all subpopulations"
        best_solution = None
        best_fitness = jnp.inf
        for pop in range(self.num_populations):
            if jnp.min(fitnesses[pop]) < best_fitness:
                best_fitness = jnp.min(fitnesses[pop])
                best_solution = populations[pop][jnp.argmin(fitnesses[pop])]
        return best_solution
    
    def get_root(self, tree: list):
        "Returns the type of the root of a tree"
        root = tree()[0]
        if isinstance(root, jax.numpy.ndarray):
            root = "Constant"
        return root
    
    def initialize_variables(self, env):
        "Defines environment and variables"
        self.env = env
        self.dt = env.dt
        self.obs_size = env.n_obs
        self.latent_size = env.n_var
        self.control_size = env.n_control

        #Define the variables that can be included in solutions
        self.variables = ["y" + str(i) for i in range(self.obs_size)]
        self.state_variables = ["a" + str(i) for i in range(self.state_size)]
        self.control_variables = ["u" + str(i) for i in range(self.control_size)]

    #Similarity methods
    def similarity(self, tree_a: list, tree_b: list):
        #Computes the similarity of two trees. 
        if len(tree_a)==1 and len(tree_b)==1:
            if isinstance(tree_a[0], jax.numpy.ndarray) and isinstance(tree_b[0], jax.numpy.ndarray):
                return 1
            if tree_a==tree_b:
                return 1
            return 0
        if len(tree_a) != len(tree_b):
            return 0
        if len(tree_a) == 2:
            return int(tree_a[0] == tree_b[0]) + self.similarity(tree_a[1], tree_b[1])
        #Compute the similarity for both permutations of the children nodes
        return int(tree_a[0] == tree_b[0]) + max(self.similarity(tree_a[1], tree_b[1]) + self.similarity(tree_a[2], tree_b[2]), self.similarity(tree_a[1], tree_b[2]) + self.similarity(tree_a[2], tree_b[1]))

    def similarities(self, tree_a: list, tree_b: list):
        #Accumulates the similarities for each pair of trees in two candidates
        similarity = 0
        for i in range(self.state_size):
            similarity += self.similarity(tree_a()[i],tree_b()[i])/min(len(jax.tree_util.tree_leaves(tree_a()[i])),len(jax.tree_util.tree_leaves(tree_b()[i])))
        similarity += self.similarity(tree_a.readout_tree,tree_b.readout_tree)/min(len(jax.tree_util.tree_leaves(tree_a.readout_tree)),len(jax.tree_util.tree_leaves(tree_b.readout_tree)))
        return similarity/(self.state_size+1)

    #Simplification methods
    def tree_to_sympy(self, tree: list):
        "Transforms a tree to sympy format"
        str_sol = self._to_string(tree)
        expr = sympy.parsing.sympy_parser.parse_expr(str_sol)
        if expr==sympy.nan or expr.has(sympy.core.numbers.ImaginaryUnit, sympy.core.numbers.ComplexInfinity): #Return None when the sympy expression contains illegal terms
            return None
        return expr
    
    def trees_to_sympy(self, trees: NetworkTrees):
        "Transforms trees to sympy formats"
        sympy_trees = []
        for tree in trees():
            sympy_trees.append(self.tree_to_sympy(tree))
        return sympy_trees

    def replace_negatives(self, tree: list):
        "Replaces '+-1.0*x' with '-x' to simplify trees even further"
        if len(tree)<3:
            return tree
        left_tree = self.replace_negatives(tree[1])
        right_tree = self.replace_negatives(tree[2])

        if tree[0]=="+" and right_tree[0]=="*" and right_tree[1]==-1.0:
            return ["-", left_tree, right_tree[2]]
        return [tree[0], left_tree, right_tree]

    def sympy_to_tree(self, sympy_expr, mode: str):
        "Reconstruct a tree from a sympy expression"
        if isinstance(sympy_expr,sympy.Float) or isinstance(sympy_expr,sympy.Integer):
            return [jnp.array(float(sympy_expr))]
        elif isinstance(sympy_expr,sympy.Symbol):
            return [str(sympy_expr)]
        elif isinstance(sympy_expr,sympy.core.numbers.NegativeOne):
            return [jnp.array(-1)]
        elif isinstance(sympy_expr,sympy.core.numbers.Zero):
            return [jnp.array(0)]
        elif isinstance(sympy_expr,sympy.core.numbers.Half):
            return [jnp.array(0.5)]
        elif isinstance(sympy_expr*-1, sympy.core.numbers.Rational):
            return [jnp.array(float(sympy_expr))]
        elif not isinstance(sympy_expr,tuple):
            if isinstance(sympy_expr,sympy.Add):
                left_tree = self.sympy_to_tree(sympy_expr.args[0], "Add")
                if len(sympy_expr.args)>2:
                    right_tree = self.sympy_to_tree(sympy_expr.args[1:], "Add")
                else:
                    right_tree = self.sympy_to_tree(sympy_expr.args[1], "Add")
                return ["+",left_tree,right_tree]
            if isinstance(sympy_expr,sympy.Mul):
                left_tree = self.sympy_to_tree(sympy_expr.args[0], "Mul")
                if len(sympy_expr.args)>2:
                    right_tree = self.sympy_to_tree(sympy_expr.args[1:], "Mul")
                else:
                    right_tree = self.sympy_to_tree(sympy_expr.args[1], "Mul")
                return ["*",left_tree,right_tree]
            if isinstance(sympy_expr,sympy.cos):
                return ["cos",self.sympy_to_tree(sympy_expr.args[0], mode=None)]
            if isinstance(sympy_expr,sympy.sin):
                return ["sin",self.sympy_to_tree(sympy_expr.args[0], mode=None)]
            if isinstance(sympy_expr, sympy.Pow):
                if sympy_expr.args[1]==-1:
                    right_tree = self.sympy_to_tree(sympy_expr.args[0], "Mul")
                    return ["/",[jnp.array(1)],right_tree]
                else:
                    left_tree = self.sympy_to_tree(sympy_expr.args[0], "Add")
                    right_tree = self.sympy_to_tree(sympy_expr.args[1], "Add")
                    return ["**", left_tree, right_tree]
        else:
            if mode=="Add":
                left_tree = self.sympy_to_tree(sympy_expr[0], "Add")
                if len(sympy_expr)>2:
                    right_tree = self.sympy_to_tree(sympy_expr[1:], "Add")
                else:
                    right_tree = self.sympy_to_tree(sympy_expr[1], "Add")
                return ["+",left_tree,right_tree]
            if mode=="Mul":
                left_tree = self.sympy_to_tree(sympy_expr[0], "Mul")
                if len(sympy_expr)>2:
                    right_tree = self.sympy_to_tree(sympy_expr[1:], "Mul")
                else:
                    right_tree = self.sympy_to_tree(sympy_expr[1], "Mul")
                return ["*",left_tree,right_tree]
        
    def simplify_tree(self, tree: list):
        "Simplifies a tree by transforming the tree to sympy format and reconstructing it"
        expr = self.tree_to_sympy(tree)
        if expr == None:
            return None
        new_tree = self.sympy_to_tree(expr, "Add" * isinstance(expr, sympy.Add) + "Mul" * isinstance(expr, sympy.Mul))
        new_tree = self.replace_negatives(new_tree)
        return new_tree

    #Migration methods
    def migrate_trees(self, sender: list, receiver: list):
        "Selects individuals that will replace randomly selected indivuals in a receiver distribution"
        sender_distribution = 1/jnp.array([p.fitness for p in sender]) #Select fitter individuals to send with higher probability
        sender_indices = jrandom.choice(self.keyGen.get(), self.population_indices, shape=(self.migration_size,), p=sender_distribution, replace=False)

        receiver_distribution = jnp.array([p.fitness for p in receiver]) #Select unfit individuals to replace with higher probability
        receiver_indices = jrandom.choice(self.keyGen.get(), self.population_indices, shape=(self.migration_size,), p=receiver_distribution, replace=False)

        new_population = receiver

        for i in range(self.migration_size):
            new_population[receiver_indices[i]] = sender[sender_indices[i]]

        return new_population

    def migrate_populations(self, populations: list):
        "Manages the migration between pairs of populations"
        assert (self.migration_method=="ring") or (self.migration_method=="random"), "This method is not implemented"
        if self.num_populations==1: #No migration possible
            return populations

        populations_copy = populations

        if self.migration_method=="ring":
            for pop in range(self.num_populations):
                #Determine destination and sender
                destination = populations_copy[pop]
                sender = populations_copy[(pop+1)%self.num_populations]
                populations[pop] = self.migrate_trees(sender, destination)
        
        elif self.migration_method=="random":
            permutation = jrandom.permutation(self.keyGen.get(), jnp.arange(self.num_populations))
            for pop in range(self.num_populations):
                #Determine destination and sample sender
                destination = populations_copy[pop]
                sender = populations_copy[permutation[pop]]
                populations[pop] = self.migrate_trees(sender, destination)

        return populations

    #Initialization methods
    def sample_leaf(self, sd: float = 1.0):
        "Samples a random leaf. The leaf is either a contant or a variable"
        leaf_type = jrandom.uniform(self.keyGen.get(), shape=())
        if leaf_type<0.4:
            return [sd*jrandom.normal(self.keyGen.get())] #constant
        elif leaf_type<0.6: 
            return [self.variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.variables))]] #observed variable
        elif leaf_type<0.8:
            return [self.state_variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.state_variables))]] #hidden state variable
        elif leaf_type<0.9:
            return [self.control_variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.control_variables))]] #control variable
        else:
            return ['target'] #target
    
    def grow_node(self, depth: int):
        "Generates a random node that can contain leaves at lower depths"
        if depth == 1: #If depth is reached, a leaf is sampled
            return self.sample_leaf(sd=3)
        
        leaf_type = jrandom.choice(self.keyGen.get(), jnp.arange(3), p=jnp.array([0.3,0.0,0.7])) #Sample the type of the leaf #NO UNARY OPERATOR NODES
        tree = []
        if leaf_type == 0: #leaf
            return self.sample_leaf(sd=3)
        elif leaf_type == 1: #unary operator
            tree.append(self.unary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.unary_operators))])
            tree.append(self.grow_node(depth-1))
            return tree
        elif leaf_type == 2: #binary operator
            tree.append(self.binary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.binary_operators))])
            tree.append(self.grow_node(depth-1))
            tree.append(self.grow_node(depth-1))
            return tree
        
    def full_node(self, depth: int):
        "Generates a random node that can only have leaves at the highest depth"
        if depth == 1: #If depth is reached, a leaf is sampled
            return self.sample_leaf(sd=3)
        leaf_type = jrandom.uniform(self.keyGen.get(), shape=()) #Sample the type of the leaf #NO UNARY OPERATOR NODES
        tree = []
        if leaf_type > 1.0: #unary operator
            tree.append(self.unary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.unary_operators))])
            tree.append(self.full_node(depth-1))
            return tree
        else: #binary operator
            tree.append(self.binary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.binary_operators))])
            tree.append(self.full_node(depth-1))
            tree.append(self.full_node(depth-1))
            return tree

    def sample_readout_leaf(self, sd: float = 1.0):
        "Samples a random leaf for the readout tree. The leaf is either a variable or a constant. Observations and control variables are excluded"
        leaf_type = jrandom.uniform(self.keyGen.get(), shape=())
        if leaf_type<0.4:
            return [sd*jrandom.normal(self.keyGen.get())] #constant
        elif leaf_type<0.9:
            return [self.state_variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.state_variables))]] #Hidden state variable
        else:
            return ['target'] #Target

    def sample_readout_tree(self, depth: int):
        "Generates a random node for the readout tree that can contain leaves at lower depths"
        if depth == 1:
            return self.sample_readout_leaf(sd=3) #If depth is reached, a leaf is sampled
        leaf_type = jrandom.choice(self.keyGen.get(), jnp.arange(3), p=jnp.array([0.3,0.0,0.7])) #u=0.3
        tree = []
        if leaf_type == 0: #leaf
            return self.sample_readout_leaf(sd=3)
        elif leaf_type == 1: #unary operator
            tree.append(self.unary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.unary_operators))])
            tree.append(self.sample_readout_tree(depth-1))
            return tree
        elif leaf_type == 2: #binary operator
            tree.append(self.binary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.binary_operators))])
            tree.append(self.sample_readout_tree(depth-1))
            tree.append(self.sample_readout_tree(depth-1))
            return tree

    def sample_trees(self, max_depth: int = 3, N: int = 1, num_populations: int = 1, init_method: str = "ramped"):
        "Samples multiple populations of models with a certain (max) depth given a specified method"
        assert (init_method=="ramped") or (init_method=="full") or (init_method=="grow"), "This method is not implemented"

        populations = []
        if init_method=="grow": #Allows for leaves at lower depths
            for _ in range(num_populations):
                population = []
                while len(population) < N:
                    depth = jrandom.randint(self.keyGen.get(), (), 2, max_depth+1)
                    trees = []
                    for _ in range(self.state_size):
                        trees.append(self.grow_node(depth))
                    readout = self.sample_readout_tree(depth)
                    new_individual = NetworkTrees(trees, readout)
                    if new_individual not in population:
                        population.append(new_individual)
                populations.append(population)
        elif init_method=="full": #Does not allow for leaves at lower depths
            for _ in range(num_populations):
                population = []
                while len(population) < N:
                    depth = jrandom.randint(self.keyGen.get(), (), 2, max_depth+1)
                    trees = []
                    for _ in range(self.state_size):
                        trees.append(self.full_node(depth))
                    readout = self.sample_readout_tree(depth)
                    new_individual = NetworkTrees(trees, readout)
                    if new_individual not in population:
                        population.append(new_individual)
                populations.append(population)
        elif init_method=="ramped": #Mixes full and grow initialization, as well as different max depths
            for _ in range(num_populations):
                population = []
                while len(population) < N:
                    depth = jrandom.randint(self.keyGen.get(), (), 2, max_depth+1)
                    trees = []
                    for _ in range(self.state_size):                                
                        if jrandom.uniform(self.keyGen.get())>0.5:
                            trees.append(self.full_node(depth))
                        else:
                            trees.append(self.grow_node(depth))
                    readout = self.sample_readout_tree(depth)
                    new_individual = NetworkTrees(trees, readout)
                    if new_individual not in population:
                        population.append(new_individual)
                populations.append(population)

        if N == 1: #Return only one tree
            return populations[0][0]      
        return populations

    #Cross-over methods
    def tree_intersection(self, tree_a: list, tree_b: list, path: list = [], interior_nodes: list = [], boundary_nodes: list = []):
        "Determines the intersecting nodes of a pair of trees. Specifies interior nodes with same arity and boundary nodes with different arity"
        if (len(tree_a) == len(tree_b)) and len(tree_a) > 1: #Check if same arity but not a leaf
            interior_nodes.append(path + [0])
            if len(tree_b) == 3:
                interior_nodes, boundary_nodes = self.tree_intersection(tree_a[1], tree_b[1], path + [1], interior_nodes, boundary_nodes)
                interior_nodes, boundary_nodes = self.tree_intersection(tree_a[2], tree_b[2], path + [2], interior_nodes, boundary_nodes)
            else:
                interior_nodes, boundary_nodes = self.tree_intersection(tree_a[1], tree_b[1], path + [1], interior_nodes, boundary_nodes)
        else:
            boundary_nodes.append(path + [0])
        
        return interior_nodes, boundary_nodes
    
    def get_subtree(self, tree: list):
        "Return the subtree from a random node onwards"
        leaves = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        distribution = jnp.array([0.5+1.5*self._is_operator(leaf) for leaf in leaves]) #increase selection probability of operators
        distribution = distribution.at[0].set(0.5) #lower probability of root node
        index = jrandom.choice(self.keyGen.get(), jnp.arange(len(leaves)),p=distribution)

        path = flat_tree_and_path[index][0][:-1]

        subtree = self._key_loc(tree, path)

        return path, subtree
    
    def standard_cross_over(self, trees_a: NetworkTrees, trees_b: NetworkTrees):
        "Performs standard cross-over on a pair of trees, returning two new trees. A cross-over point is selected in both trees, interchanging the subtrees below this point"
        for i in range(self.state_size):
            path_a, subtree_a = self.get_subtree(trees_a()[i])
            path_b, subtree_b = self.get_subtree(trees_b()[i])

            trees_a = eqx.tree_at(lambda t: self._key_loc(t()[i], path_a), trees_a, subtree_b)
            trees_b = eqx.tree_at(lambda t: self._key_loc(t()[i], path_b), trees_b, subtree_a)

        #Apply cross-over to readout trees as well
        path_a, subtree_a = self.get_subtree(trees_a.readout_tree)
        path_b, subtree_b = self.get_subtree(trees_b.readout_tree)

        trees_a = eqx.tree_at(lambda t: self._key_loc(t.readout_tree, path_a), trees_a, subtree_b)
        trees_b = eqx.tree_at(lambda t: self._key_loc(t.readout_tree, path_b), trees_b, subtree_a)

        return trees_a, trees_b
    
    def tree_cross_over(self, tree_a: list, tree_b: list):
        "Applies cross-over on tree level, interchanging only full trees"
        new_tree_a = tree_a
        new_tree_b = tree_b
        for i in range(self.state_size):
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: t()[i], new_tree_a, tree_b()[i])
                new_tree_b = eqx.tree_at(lambda t: t()[i], new_tree_b, tree_a()[i])
        if jrandom.uniform(self.keyGen.get()) > 0.5: #Apply cross-over to readout
            new_tree_a = eqx.tree_at(lambda t: t.readout_tree, new_tree_a, tree_b.readout_tree)
            new_tree_b = eqx.tree_at(lambda t: t.readout_tree, new_tree_b, tree_a.readout_tree)
        return new_tree_a, new_tree_b

    def uniform_cross_over(self, tree_a: list, tree_b: list):
        "Performs uniform cross-over on a pair of trees, returning two new trees. Each overlapping node is switched with 50% chance and children of boundary nodes are switched as well."
        for i in range(self.state_size): #Get intersection of the trees
            interior_nodes, boundary_nodes = self.tree_intersection(tree_a()[i], tree_b()[i], path = [], interior_nodes = [], boundary_nodes = [])
            new_tree_a = tree_a
            new_tree_b = tree_b

            for node in interior_nodes: #Randomly switch two nodes of interior intersecting nodes
                if jrandom.uniform(self.keyGen.get()) > 0.5:
                    new_tree_a = eqx.tree_at(lambda t: self._index_loc(t()[i], node), new_tree_a, self._index_loc(tree_b()[i], node))
                    new_tree_b = eqx.tree_at(lambda t: self._index_loc(t()[i], node), new_tree_b, self._index_loc(tree_a()[i], node))

            for node in boundary_nodes: #Randomly switch two nodes and their children of boundary intersecting nodes
                if jrandom.uniform(self.keyGen.get()) > 0.5:
                    new_tree_a = eqx.tree_at(lambda t: self._index_loc(t()[i], node[:-1]), new_tree_a, self._index_loc(tree_b()[i], node[:-1]))
                    new_tree_b = eqx.tree_at(lambda t: self._index_loc(t()[i], node[:-1]), new_tree_b, self._index_loc(tree_a()[i], node[:-1]))
            
            tree_a = new_tree_a
            tree_b = new_tree_b

        #Apply cross-over to readout
        interior_nodes, boundary_nodes = self.tree_intersection(tree_a.readout_tree, tree_b.readout_tree, path = [], interior_nodes = [], boundary_nodes = [])
        new_tree_a = tree_a
        new_tree_b = tree_b

        #Randomly switch two nodes of interior intersecting nodes
        for node in interior_nodes:
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: self._index_loc(t.readout_tree, node), new_tree_a, self._index_loc(tree_b.readout_tree, node))
                new_tree_b = eqx.tree_at(lambda t: self._index_loc(t.readout_tree, node), new_tree_b, self._index_loc(tree_a.readout_tree, node))
        #Randomly switch two nodes and their children of boundary intersecting nodes
        for node in boundary_nodes:
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: self._index_loc(t.readout_tree, node[:-1]), new_tree_a, self._index_loc(tree_b.readout_tree, node[:-1]))
                new_tree_b = eqx.tree_at(lambda t: self._index_loc(t.readout_tree, node[:-1]), new_tree_b, self._index_loc(tree_a.readout_tree, node[:-1]))
        
        tree_a = new_tree_a
        tree_b = new_tree_b

        return tree_a, tree_b
    
    #Mutation methods
    def insert_operator(self, tree: list, readout: bool = False):
        "Insert an operator at a random point in tree. Sample a new leaf if necessary to satisfy arity of the operator"
        nodes = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        operator_indices = jnp.ravel(jnp.argwhere(jnp.array([self._is_operator(node) for node in nodes])))
        index = jrandom.choice(self.keyGen.get(), operator_indices)
        path = flat_tree_and_path[index][0][:-1]
        subtree = self._key_loc(tree, path)
        
        operator_type = jrandom.uniform(self.keyGen.get())
        if operator_type>1.0: #unary operator
            new_operator = self.unary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.unary_operators))]
            new_tree = [new_operator, subtree]
        else: #binary operator
            new_operator = self.binary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.binary_operators))]
            tree_position = jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=2)
            #Sample leaf for other child of operator
            if readout:
                other_leaf = self.sample_readout_leaf()
            else:
                other_leaf = self.sample_leaf()

            new_tree = [new_operator, subtree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, subtree]
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_tree)
    
    def add_subtree(self, tree: list, readout: bool = False):
        #Replace a leaf with a new subtree
        nodes = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([self._is_leaf(node) for node in nodes])))
        index = jrandom.choice(self.keyGen.get(), leaf_indices)
        path = flat_tree_and_path[index][0][:-1]
        if readout:
            return eqx.tree_at(lambda t: self._key_loc(t, path), tree, self.sample_readout_tree(depth=3))
        else:
            return eqx.tree_at(lambda t: self._key_loc(t, path), tree, self.grow_node(depth=3))
    
    def mutate_operator(self, tree: list):
        "Replace an operator with different operator of equal arity"
        nodes = jax.tree_util.tree_leaves(tree)
        operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._is_operator(node) for node in nodes])))
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        index = jrandom.choice(self.keyGen.get(), operator_indicies)
        symbol = flat_tree_and_path[index][1]
        path = flat_tree_and_path[index][0]

        if symbol in self.binary_operators:
            bin_copy = self.binary_operators.copy()
            bin_copy.remove(symbol)
            new_operator = bin_copy[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(bin_copy))]
            new_tree = eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_operator)
        elif symbol in self.unary_operators:
            un_copy = self.unary_operators.copy()
            un_copy.remove(symbol)
            new_operator = un_copy[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(un_copy))]
            new_tree = eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_operator)
        return eqx.tree_at(lambda t: t, tree, new_tree)

    def prepend_operator(self, tree: list, readout: bool = False):
        "Add an operator to the top of the tree"
        if jrandom.uniform(self.keyGen.get())>1.0:
            new_operator = self.unary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.unary_operators))]
            new_tree = [new_operator, tree]
        else:
            new_operator = self.binary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.binary_operators))]
            tree_position = jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=2)
            #Sample a leaf for the other child of the operator
            if readout:
                other_leaf = self.sample_readout_leaf()
            else:
                other_leaf = self.sample_leaf()

            new_tree = [new_operator, tree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, tree]
        return eqx.tree_at(lambda t: t, tree, new_tree)

    def mutate_leaf(self, tree: list, readout: bool = False):
        "Change value of a leaf. Leaf can stay the same type of change to a different leaf type"
        nodes = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([self._is_leaf(node) for node in nodes])))
        index = jrandom.choice(self.keyGen.get(), leaf_indices)
        new_leaf = [nodes[index]]
        while new_leaf == [nodes[index]]:
            index = jrandom.choice(self.keyGen.get(), leaf_indices)
            if readout:
                new_leaf = self.sample_readout_leaf(sd=3)
            else:
                new_leaf = self.sample_leaf(sd=3)
        path = flat_tree_and_path[index][0][:-1]
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_leaf)
    
    def mutate_constant(self, tree: list):
        "Change the value of a constant leaf. The value is sampled close to the old value"
        nodes = jax.tree_util.tree_leaves(tree)
        constant_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._is_constant(node) for node in nodes])))
        index = jrandom.choice(self.keyGen.get(), constant_indicies)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        value = flat_tree_and_path[index][1]
        path = flat_tree_and_path[index][0]
        #Sample with old value as mean
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, value+jrandom.normal(self.keyGen.get()))
    
    def delete_operator(self, tree: list, readout: bool = False):
        "Replace an operator with a new leaf"
        if readout:
            new_leaf = self.sample_readout_leaf(sd=3)
        else:
            new_leaf = self.sample_leaf(sd=3)
        nodes = jax.tree_util.tree_leaves(tree)
        operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._is_operator(node) for node in nodes])))
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        index = jrandom.choice(self.keyGen.get(), operator_indicies)
        path = flat_tree_and_path[index][0][:-1]

        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_leaf)

    def mutate_tree(self, tree: list, allow_simplification: bool = True, readout: bool = False):
        #Applies on of the mutation types to a tree
        mutation_probabilities = self.mutation_probabilities.copy()

        if len(tree)==1: #Tree does not contain operators, so exclude mutations that require operators
            mutation_probabilities = mutation_probabilities.at[self.mutation_types.index("mutate_operator")].set(0)
            mutation_probabilities = mutation_probabilities.at[self.mutation_types.index("delete_operator")].set(0)
            mutation_probabilities = mutation_probabilities.at[self.mutation_types.index("insert_operator")].set(0)
        if sum([self._is_constant(node) for node in jax.tree_util.tree_leaves(tree)]) == 0: #Tree does not contain constants, so exclude mutations that require constants
            mutation_probabilities = mutation_probabilities.at[self.mutation_types.index("mutate_constant")].set(0)
        if not allow_simplification: #Simplifying the tree is not possible, so exclude simplification
            mutation_probabilities = mutation_probabilities.at[self.mutation_types.index("simplify_tree")].set(0)

        mutation_type = self.mutation_types[jrandom.choice(self.keyGen.get(), jnp.arange(len(self.mutation_types)), shape=(), p=mutation_probabilities)]
        if mutation_type=="mutate_operator":
            new_tree = self.mutate_operator(tree)
        elif mutation_type=="delete_operator":
            new_tree = self.delete_operator(tree, readout)
        elif mutation_type=="prepend_operator":
            new_tree = self.prepend_operator(tree, readout)
        elif mutation_type=="insert_operator":
            new_tree = self.insert_operator(tree, readout)
        elif mutation_type=="mutate_constant":
            new_tree = self.mutate_constant(tree)
        elif mutation_type=="mutate_leaf":
            new_tree = self.mutate_leaf(tree, readout)
        elif mutation_type=="sample_subtree":
            if readout:
                new_tree = self.sample_readout_tree(depth=self.max_init_depth)
            else:
                new_tree = self.grow_node(depth=self.max_init_depth)
        elif mutation_type=="add_subtree":
            new_tree = self.add_subtree(tree, readout)
        elif mutation_type=="simplify_tree":
            old_tree_string = sympy.parsing.sympy_parser.parse_expr(self._to_string(tree),evaluate=False)
            new_tree_string = sympy.parsing.sympy_parser.parse_expr(self._to_string(tree))
            if old_tree_string==new_tree_string: #Simplification does not change the tree
                new_tree = self.mutate_tree(tree, allow_simplification=False, readout=readout)
            
            simplified_tree = self.simplify_tree(tree)
            if simplified_tree==None: #Simplification gives an illegal expression
                new_tree = self.mutate_tree(tree, allow_simplification=False, readout=readout)
            else:
                new_tree = simplified_tree
        if (self._tree_depth(new_tree) > self.max_depth): #If the new tree exceeds the max depth, apply mutation to the original tree again
            new_tree = self.mutate_tree(tree,readout=readout)
        return new_tree

    #Selection and reproduction methods
    def tournament_selection(self, population: list, population_index: int):
        "Selects a candidate from a randomly selected tournament. Selection is based on fitness and the probability of being chosen given a rank"
        tournament = []
        #Sample solutions to include in the tournament
        tournament_indices = jrandom.choice(self.keyGen.get(), self.population_indices, shape=(self.tournament_size,), replace=False)
        for i in tournament_indices:
            tournament.append(population[i])
        #Sort on fitness
        tournament.sort(key=lambda x: x.fitness)
        #Sample tournament winner
        index = jrandom.choice(self.keyGen.get(), self.tournament_indices, p=self.tournament_probabilities[population_index])
        return tournament[index]
    
    def next_population(self, population: list, mean_fitness: float, population_index: int):
        "Generates a new population by evolving the current population. After cross-over and mutation, the new trees are checked to be different from their parents."
        remaining_candidates = len(population)
        new_pop = []
        failed_mutations = 0

        while remaining_candidates>1: #Loop until new population has reached the desired size
            probs = self.reproduction_probabilities[population_index]
            tree_a = self.tournament_selection(population, population_index)
            tree_b = self.tournament_selection(population, population_index)

            similarity = self.similarities(tree_a, tree_b)
            if tree_a.fitness > mean_fitness and tree_b.fitness > mean_fitness:
                probs = [0,0.5,0.5,0] #Do not apply cross-over if trees have poor fitness

            elif similarity > self.similarity_threshold:
                probs = [0,0.6,0.3,0.1] #Do not apply crossover if trees are similar

            reproduction_type = jrandom.choice(self.keyGen.get(), jnp.arange(4), p=jnp.array(probs))
                
            if reproduction_type==0: #Cross-over
                cross_over_type = jrandom.uniform(self.keyGen.get()) #Sample a cross-over method           
                if cross_over_type < 0.3:
                    new_tree_a, new_tree_b = self.tree_cross_over(tree_a, tree_b)
                elif cross_over_type < 0.7:
                    new_tree_a, new_tree_b = self.uniform_cross_over(tree_a, tree_b)
                else:
                    new_tree_a, new_tree_b = self.standard_cross_over(tree_a, tree_b)

                #If both trees remain the same or one of the trees exceeds the max depth, cross-over has failed
                if eqx.tree_equal(tree_a, new_tree_a) and eqx.tree_equal(tree_b, new_tree_b):
                    failed_mutations += 1
                elif (self._tree_depth(new_tree_a) > self.max_depth) or (self._tree_depth(new_tree_b) > self.max_depth):
                    failed_mutations += 1
                else:
                    #Append new trees to the new population
                    new_pop.append(new_tree_a)
                    new_pop.append(new_tree_b)
                    remaining_candidates -= 2

            elif reproduction_type==1: #Mutation
                mutate_bool = jrandom.uniform(self.keyGen.get(), shape=(self.state_size,))
                while not any(mutate_bool>self.tree_mutate_probs[population_index]): #Make sure that at least one tree is mutated
                    mutate_bool = jrandom.uniform(self.keyGen.get(), shape=(self.state_size,))

                new_tree_a = tree_a
                for i in range(self.state_size):
                    if mutate_bool[i]>self.tree_mutate_probs[population_index]:
                        new_tree = self.mutate_tree(tree_a()[i])
                        new_tree_a = eqx.tree_at(lambda t: t()[i], new_tree_a, new_tree)
                if jrandom.uniform(self.keyGen.get()) > self.tree_mutate_probs[population_index]: #Mutate readout tree
                    new_tree_a = eqx.tree_at(lambda t: t.readout_tree, new_tree_a, self.mutate_tree(new_tree_a.readout_tree, readout=True))
                #Add new tree to the new population
                new_pop.append(new_tree_a)
                remaining_candidates -= 1

                mutate_bool = jrandom.uniform(self.keyGen.get(), shape=(self.state_size,))
                while not any(mutate_bool>self.tree_mutate_probs[population_index]): #Make sure that at least one tree is mutated
                    mutate_bool = jrandom.uniform(self.keyGen.get(), shape=(self.state_size,))

                new_tree_b = tree_b
                for i in range(self.state_size):
                    if mutate_bool[i]>self.tree_mutate_probs[population_index]:
                        new_tree = self.mutate_tree(tree_b()[i])
                        new_tree_b = eqx.tree_at(lambda t: t()[i], new_tree_b, new_tree)
                if jrandom.uniform(self.keyGen.get()) > self.tree_mutate_probs[population_index]: #Mutate readout tree
                    new_tree_b = eqx.tree_at(lambda t: t.readout_tree, new_tree_b, self.mutate_tree(new_tree_b.readout_tree, readout=True))
                #Add new tree to the new population
                new_pop.append(new_tree_b)
                remaining_candidates -= 1

            elif reproduction_type==2: #Sample new trees
                new_trees = self.sample_trees(max_depth=self.max_init_depth, N=2, init_method="full")[0]
                #Add new trees to the new population
                remaining_candidates -= 2
                new_pop.append(new_trees[0])
                new_pop.append(new_trees[1])
            elif reproduction_type==3: #Reproduction
                remaining_candidates -= 2
                #Add new trees to the new population
                new_pop.append(tree_a)
                new_pop.append(tree_b)
        
        best_candidate = sorted(population, key=lambda x: x.fitness, reverse=False)[0] #Keep best candidate in new population
        if remaining_candidates==0:
            new_pop[0] = best_candidate
        else:
            new_pop.append(best_candidate)
        return new_pop
   
    #Evaluation methods
    def evaluate_tree(self, tree: list):
        "A tree is transformed to a callable function, represented as nested lamba functions"
        if tree[0] in self.binary_operators:
            assert len(tree) == 3, f"The operator {tree[0]} requires two inputs"
            left = self.evaluate_tree(tree[1])
            right = self.evaluate_tree(tree[2])
            return lambda x, a, u, t: self.binary_operators_map[tree[0]](left(x,a,u,t),right(x,a,u,t))
        
        elif tree[0] in self.unary_operators:
            assert len(tree) == 2, f"The operator {tree[0]} requires one input"
            left = self.evaluate_tree(tree[1])
            return lambda x, a, u, t: self.unary_operators_map[tree[0]](left(x,a,u,t))

        assert len(tree) == 1, "Leaves should not have children"
        if isinstance(tree[0],jax.numpy.ndarray):
            return lambda x, a, u, t: tree[0]
        elif tree[0]=="target":
            return lambda x, a, u, t: t
        elif tree[0] in self.state_variables:
            return lambda x, a, u, t: a[self.state_variables.index(tree[0])]
        elif tree[0] in self.variables:
            return lambda x, a, u, t: x[self.variables.index(tree[0])]
        elif tree[0] in self.control_variables:
            return lambda x, a, u, t: u[self.control_variables.index(tree[0])]
        print(tree)
    
    def evaluate_trees(self, trees: NetworkTrees):
        "Evaluate the trees in the network"
        return [self.evaluate_tree(tree) for tree in trees()]
   
    def evaluate_control_loop(self, model: NetworkTrees, x0: Sequence[float], ts: Sequence[float], target: float, key: PRNGKey, params: Tuple):
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation of the control is defined by the set of trees
        Inputs:
            model (NetworkTrees): Model with trees for the hidden state and readout
            x0 (float): Initial state of the system
            ts (Array[float]): time points on which the controller is evaluated
            target (float): Target position that the system should reach
            key (PRNGKey)
            params (Tuple[float]): Parameters that define the system

        Returns:
            xs (Array[float]): States of the system at every time point
            ys (Array[float]): Observations of the system at every time point
            us (Array[float]): Control of the model at every time point
            activities (Array[float]): Activities of the hidden state of the model at every time point
            fitness (float): Fitness of the model 
        """

        env = copy.copy(self.env)
        env.initialize(params)

        #Trees to callable functions
        tree_funcs = self.evaluate_trees(model)
        state_equation = jax.jit(lambda y, a, u, tar: jnp.array([tree_funcs[i](y, a, u, tar) for i in range(self.state_size)]))
        readout_layer = self.evaluate_tree(model.readout_tree)
        readout = lambda y, a, _, tar: readout_layer(y, a, _, tar)

        #Define state equation
        def _drift(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]

            # jax.debug.print("P={P}", P=a[2:])

            y = env.f_obs(t, x) #Get observations from system
            u = jnp.array([readout(y, a, 0, target)]) #Readout control from hidden state
            
            dx = env.drift(t, x, u) #Apply control to system and get system change
            da = state_equation(y, a, u, target) #Compute hidden state updates
            return jnp.concatenate([dx, da])
        
        #Define diffusion
        def _diffusion(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]
            y = env.f_obs(t, x)
            u = jnp.array([readout(y, a, 0, target)])
            return jnp.concatenate([env.diffusion(t, x, u), jnp.zeros((self.state_size, 2))]) #Only the system is stochastic
        
        solver = diffrax.Euler()
        dt0 = 0.005
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])
        # _x0 = jnp.concatenate([x0, jnp.zeros(2), jnp.array([1,0,0,1])*self.env.obs_noise])

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(2,), key=key)
        system = diffrax.MultiTerm(diffrax.ODETerm(_drift), diffrax.ControlTerm(_diffusion, brownian_motion))

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=16**4
        )

        xs = sol.ys[:,:self.latent_size]
        ys = jax.vmap(env.f_obs)(ts, xs) #Map states to observations
        activities = sol.ys[:,self.latent_size:]
        us = jax.vmap(readout, in_axes=[0,0,None, None])(ys, activities, jnp.array([0]), target) #Map hidden state to control
        fitness = env.fitness_function(xs, us, target) #Compute fitness with cost function in the environment
        return xs, ys, us, activities, fitness
    
    def get_fitness(self, model: NetworkTrees, data: Tuple, add_regularization: bool = True):
        "Determine the fitness of a tree by simulating the environment and controller as a coupled system"
        x0, ts, targets, noise_keys, params = data
        _, _, _, _, fitness = jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0])(model, x0, ts, targets, noise_keys, params) #Run coupled differential equations of state and control and get fitness of the model

        fitness = jnp.mean(fitness[:,-1])*self.dt
        
        if jnp.isinf(fitness) or jnp.isnan(fitness):
            fitness = jnp.array(1e6)
        if add_regularization: #Add regularization to punish trees for their size
            return jnp.clip(fitness + self.parsimony_punishment*len(jax.tree_util.tree_leaves(model)),0,1e6)
        else:
            return jnp.clip(fitness,0,1e6)
    
    #Main algorithms
    def random_search(self, env, data, num_generations, pool_size = 10, continue_population=None, converge_value=0.0):
        """
        Applies random search to find the best solution. At every generation a new population is sampled and the best solution is kept alive.
        Inputs:
            env (Class): Environment to evaluate the solutions on.
            data (Tuple): Contains the data necessary to simulate the system. Includes initial states of the environment, time points, keys, targets and parameters describing the environment
            num_generations (int): Number of generations to run the algorithm
            pool_size (int): Size of worker pool to parallelize solution evaluation
            continue_population (list): Either start from a new population or continue from an existing population
            converge_value (float): Value that determines whether a solution in the population achieved satisfactory results

        Returns:
            best_fitnesses (Array[float]): Best fitness at every generation
            best_solution (list): Best solution at the end of the run
            populations (list): Final population that can be used to continue a run from
        """

        pool = Pool(pool_size)
        pool.close()

        self.initialize_variables(env)

        best_fitnesses = jnp.zeros(num_generations)
        best_solutions = []
        if continue_population is None:
            #Initialize new population
            populations = self.sample_trees(self.max_depth, self.population_size, num_populations=self.num_populations, init_method=self.init_method)
        else:
            #Continue from a previous population
            populations = continue_population
        
        for g in range(num_generations):
            pool.restart()
            fitness = pool.amap(lambda x: self.get_fitness(x,data),self.flatten(populations)) #Evaluate each solution parallely on a pool of workers
            pool.close()
            pool.join()
            tries = 0
            while not fitness.ready():
                time.sleep(1)
                tries += 1

                if tries >= 200:
                    print("TIMEOUT")
                    break

            flat_fitnesses = jnp.array(fitness.get())
            fitnesses = jnp.reshape(flat_fitnesses,(self.num_populations,self.population_size))
            
            #Set the fitness of each solution
            for pop in range(self.num_populations):
                population = populations[pop]
                for candidate in range(self.population_size):
                    population[candidate].set_fitness(fitnesses[pop,candidate])
            best_fitnesses = best_fitnesses.at[g].set(self.get_fitness(self._best_solution(populations, fitnesses), data, add_regularization=False))

            if best_fitnesses[g]<converge_value: #A solution reached a satisfactory score
                best_solution = self._best_solution(populations, fitnesses)
                best_solution_string = self.trees_to_sympy(best_solution)
                best_solutions.append(best_solution)
                print(f"Converge settings satisfied, best fitness {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout_tree)}")
                best_fitnesses = best_fitnesses.at[g:].set(best_fitnesses[g])

                break

            elif g < num_generations-1: #The final generation has not been reached yet, so a new population is sampled
                best_solution = self._best_solution(populations, fitnesses)
                best_solution_string = self.trees_to_sympy(best_solution)
                print(f"In generation {g+1}, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout_tree)}")
                              
                best_candidates = []
                for pop in range(self.num_populations):
                    best_candidates.append(sorted(populations[pop], key=lambda x: x.fitness, reverse=False)[0])
                populations = self.sample_trees(self.max_depth, self.population_size-1, num_populations=self.num_populations, init_method="grow")
                for pop in range(self.num_populations):
                    populations[pop].append(best_candidates[pop])
                
            else: #Final generation is reached
                best_solution = self._best_solution(populations, fitnesses)
                best_solution_string = self.trees_to_sympy(best_solution)
                best_solutions.append(best_solution)
                print(f"Final generation, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout_tree)}")

        return best_fitnesses, best_solutions, populations

    #Runs the GP algorithm for a given number of runs and generations. It is possible to continue on a previous population or start from a new population.
    def run(self, env, data, num_generations, pool_size, continue_population=None, converge_value=0, insert_solution=None):
        """
        Runs the genetic programming algorithm to find the best solution. New solutions are proposed by applying evolutionary operations to fit solutions.
        Inputs:
            env (Class): Environment to evaluate the solutions on.
            data (Tuple): Contains the data necessary to simulate the system. Includes initial states of the environment, time points, keys, targets and parameters describing the environment
            num_generations (int): Number of generations to run the algorithm
            pool_size (int): Size of worker pool to parallelize solution evaluation
            continue_population (list): Either start from a new population or continue from an existing population
            converge_value (float): Value that determines whether a solution in the population achieved satisfactory results

        Returns:
            best_fitnesses (Array[float]): Best fitness at every generation
            best_solution (list): Best solution at the end of the run
            populations (list): Final population that can be used to continue a run from
        """

        pool = Pool(pool_size)
        pool.close()

        self.initialize_variables(env)

        best_fitnesses = jnp.zeros(num_generations)
        best_fitness_per_population = jnp.zeros((num_generations, self.num_populations))
        best_solutions = []
        if continue_population is None:
            #Initialize new population
            populations = self.sample_trees(self.max_depth, self.population_size, num_populations=self.num_populations, init_method=self.init_method)
        else:
            #Continue from a previous population
            populations = continue_population

        if insert_solution is not None:
            populations[0][0] = insert_solution
        
        for g in range(num_generations):
            pool.restart()
            fitness = pool.amap(lambda x: self.get_fitness(x,data),self.flatten(populations)) #Evaluate each solution parallely on a pool of workers
            pool.close()
            pool.join()
            tries = 0
            while not fitness.ready():
                time.sleep(1)
                tries += 1

                if tries >= 200:
                    print("TIMEOUT")
                    break

            flat_fitnesses = jnp.array(fitness.get())
            fitnesses = jnp.reshape(flat_fitnesses,(self.num_populations,self.population_size))

            best_fitness_per_population = best_fitness_per_population.at[g].set(jnp.min(fitnesses, axis=1))
            
            #Set the fitness of each solution
            for pop in range(self.num_populations):
                population = populations[pop]
                for candidate in range(self.population_size):
                    population[candidate].set_fitness(fitnesses[pop,candidate])
            best_fitnesses = best_fitnesses.at[g].set(self.get_fitness(self._best_solution(populations, fitnesses), data, add_regularization=False))

            if best_fitnesses[g]<converge_value: #A solution reached a satisfactory score
                best_solution = self._best_solution(populations, fitnesses)
                best_solution_string = self.trees_to_sympy(best_solution)
                best_solutions.append(best_solution)
                print(f"Converge settings satisfied, best fitness {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout_tree)}")
                best_fitnesses = best_fitnesses.at[g:].set(best_fitnesses[g])

                break

            elif g < num_generations-1: #The final generation has not been reached yet, so a new population is sampled
                best_solution = self._best_solution(populations, fitnesses)
                best_solution_string = self.trees_to_sympy(best_solution)
                print(f"In generation {g+1}, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout_tree)}")

                #Migrate individuals between populations every few generations
                if ((g+1)%self.migration_period)==0:
                    populations = self.migrate_populations(populations)
                              
                for pop in range(self.num_populations):
                    #Generate new population
                    if g>self.restart_iter_threshold and best_fitness_per_population[g,pop] == best_fitness_per_population[g-self.restart_iter_threshold,pop]:
                        print(f"Restart in population {pop+1} at generation {g+1}")
                        populations[pop] = self.sample_trees(self.max_init_depth, self.population_size, num_populations=1, init_method=self.init_method)[0]
                    else:
                        populations[pop] = self.next_population(populations[pop], jnp.mean(flat_fitnesses), pop)
                
            else: #Final generation is reached
                best_solution = self._best_solution(populations, fitnesses)
                best_solution_string = self.trees_to_sympy(best_solution)
                best_solutions.append(best_solution)
                print(f"Final generation, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout_tree)}")

        return best_fitnesses, best_solutions, populations