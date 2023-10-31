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

class KeyGen:
    def __init__(self, seed = 0):
        self._key = jrandom.PRNGKey(seed)

    def get(self) -> jrandom.PRNGKeyArray:
        self._key, new_key = jrandom.split(self._key)
        return new_key

#Container that contains a symbolic expression. Might be redundant, but used to clearly store expressions
class Expression:
    def __init__(self, expression, readout):
        self.expression = expression
        self.readout = readout
        self.fitness = None
        self.changed = True
    
    def __call__(self):
        return self.expression
    
    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_changed(self, bool):
        self.changed = bool

    #Returns all optimisable parameters
    def get_params(self, values_and_paths):
        params = []
        paths = []
        for path, value in values_and_paths:
            if isinstance(value, jax.numpy.ndarray):
                params.append(value)
                paths.append(path)
        return jnp.array(params), paths
    
    def _to_string(self, tree):
        if len(tree)==3:
            left_tree = "(" + self._to_string(tree[1]) + ")" if len(tree[1]) == 3 else self._to_string(tree[1])
            right_tree = "(" + self._to_string(tree[2]) + ")" if len(tree[2]) == 3 else self._to_string(tree[2])
            return left_tree + tree[0] + right_tree
        elif len(tree)==2:
            return tree[0] + "(" + self._to_string(tree[1]) + ")"
        else:
            return str(tree[0])
    
    def __str__(self):
        # return f"{[self._to_string(tree) for tree in self.expression]}, readout = {str(self.readout)}"
        return f"{[self._to_string(tree) for tree in self.expression]}, readout = {self._to_string(self.readout)}"
    
#Register the container as pytree
register_pytree_node(Expression, lambda tree: ((tree.expression,tree.readout), None),
    lambda _, args: Expression(*args))

class ODE_GP:
    def __init__(self, seed, state_size, tournament_size = 5, max_depth=6, max_init_depth=3, population_size=100, num_populations=1, migration_period=5, migration_percentage=0.1, migration_method="ring", init_method="ramped", similarity_threshold=0.5):
        self.keyGen = KeyGen(seed)
        self.state_size = state_size
        self.num_populations = num_populations
        self.population_size = population_size

        #Define operators
        self.binary_operators_map = {"+":lambda a,b:a+b, "-":lambda a,b:a-b, "*":lambda a,b:a*b, "/":lambda a,b:a/b, "**":lambda a,b:a**b}
        self.unary_operators_map = {"sin":lambda a:jnp.sin(a), "cos":lambda a:jnp.cos(a)}#,"exp":lambda a:jnp.clip(jnp.exp(a),a_max=100), "sqrt":lambda a:jnp.sqrt(jnp.abs(a))}
        self.binary_operators = list(self.binary_operators_map.keys())
        self.unary_operators = list(self.unary_operators_map.keys())

        #Define modification types
        self.mutations = ["mutate_operator","delete_operator","insert_operator","mutate_constant","mutate_leaf","sample_subtree","prepend_operator","add_subtree","simplify_tree"]
        self.mod_prob = jnp.ones(len(self.mutations))
        self.mod_prob = self.mod_prob.at[self.mutations.index("mutate_operator")].set(0.5)
        self.mod_prob = self.mod_prob.at[self.mutations.index("delete_operator")].set(0.5)
        self.mod_prob = self.mod_prob.at[self.mutations.index("insert_operator")].set(1.0)
        self.mod_prob = self.mod_prob.at[self.mutations.index("mutate_constant")].set(1.0)
        self.mod_prob = self.mod_prob.at[self.mutations.index("mutate_leaf")].set(1.0)
        self.mod_prob = self.mod_prob.at[self.mutations.index("sample_subtree")].set(1)
        self.mod_prob = self.mod_prob.at[self.mutations.index("prepend_operator")].set(0.5)
        self.mod_prob = self.mod_prob.at[self.mutations.index("add_subtree")].set(1.0)
        self.mod_prob = self.mod_prob.at[self.mutations.index("simplify_tree")].set(0.5)
        
        self.tree_mutate_probs = jnp.linspace(0.5,0.2,self.num_populations)
        self.probabilities = jnp.vstack([jnp.linspace(0.1,0.8,self.num_populations),jnp.linspace(0.4,0.1,self.num_populations),jnp.linspace(0.5,0.0,self.num_populations),jnp.linspace(0.,0.1,self.num_populations)]).T
        self.restart_iter = 8

        #Specify GP settings
        self.tournament_size = tournament_size
        self.tournament_indices = jnp.arange(self.tournament_size)
        self.selection_pressures = jnp.linspace(0.6,1.0,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**self.tournament_indices for sp in self.selection_pressures])
        self.parsimony_punishment = 0.1
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.population_size_array = jnp.arange(self.population_size)
        self.migration_period = migration_period
        self.migration_size = int(migration_percentage*self.population_size)
        self.migration_method = migration_method
        self.init_method = init_method
        self.similarity_threshold = similarity_threshold

    #Returns subtree at location specified by a path
    def _key_loc(self, tree, path):
        new_tree = tree
        for k in path:
            new_tree = new_tree[k.idx]
        return new_tree
    
    #Returns subtree at location specified by indices
    def _index_loc(self, tree, path):
        new_tree = tree
        for i in path:
            new_tree = new_tree[i]
        return new_tree
    
    def _operator(self, symbol):
        return symbol in self.binary_operators or symbol in self.unary_operators
    
    def _leaf(self, symbol):
        return symbol not in self.binary_operators and symbol not in self.unary_operators
    
    def _variable(self, symbol):
        if isinstance(symbol, jax.numpy.ndarray):
            return False
        else:
            return symbol in self.variables
    
    def _constant(self, symbol):
        return isinstance(symbol, jax.numpy.ndarray)

    def _to_string(self, tree):
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
        
    #Returns highest depth of tree
    def _tree_depth(self, tree):
        flat_tree = jax.tree_util.tree_leaves_with_path(tree)
        return jnp.max(jnp.array([len(node[0]) for node in flat_tree]))
    
    #Returns depth of each node of the tree
    def _depth_per_node(self, tree):
        flat_tree, = jax.tree_util.tree_leaves_with_path(tree)
        return jnp.array([len(node[0]) for node in flat_tree])
    
    #Determines the intersecting nodes of a pair of trees. Specifies interior nodes with same arity and boundary nodes with different arity
    def _tree_intersection(self, tree1, tree2, path = [], interior_nodes = [], boundary_nodes = []):
        #Check if same arity but not a leaf
        if (len(tree1) == len(tree2)) and len(tree1) > 1:
            interior_nodes.append(path + [0])
            if len(tree2) == 3:
                interior_nodes, boundary_nodes = self._tree_intersection(tree1[1], tree2[1], path + [1], interior_nodes, boundary_nodes)
                interior_nodes, boundary_nodes = self._tree_intersection(tree1[2], tree2[2], path + [2], interior_nodes, boundary_nodes)
            else:
                interior_nodes, boundary_nodes = self._tree_intersection(tree1[1], tree2[1], path + [1], interior_nodes, boundary_nodes)
        else:
            boundary_nodes.append(path + [0])
        
        return interior_nodes, boundary_nodes
    
    def evaluate_tree(self, tree):
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
    
    def evaluate_expression(self, expression):
        return [self.evaluate_tree(tree) for tree in expression()]
    
    #Solves the coupled differential equation of the state and the control. The differential equation of the state is defined in the environment and the differential equation of the control is defined by a tree.
    def evaluate_control_loop(self, tree, x0, ts, target, key, params):
        env = copy.copy(self.env)
        env.initialize(params)
        tree_funcs = self.evaluate_expression(tree)
        state_equation = jax.jit(lambda y, a, u, tar: jnp.array([tree_funcs[i](y, a, u, tar) for i in range(self.state_size)]))
        # readout = lambda a: tree.readout@a
        readout_layer = self.evaluate_tree(tree.readout)
        readout = lambda y, a, _, tar: readout_layer(y, a, _, tar)

        #Define state equation
        def _drift(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]

            # jax.debug.print("P={P}", P=a[2:])

            y = env.f_obs(t, x)
            u = jnp.array([readout(y, a, 0, target)])
            #Get environment update
            dx = env.drift(t, x, u)
            da = state_equation(y, a, u, target)
            return jnp.concatenate([dx, da])
        
        #Define diffusion
        def _diffusion(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]
            y = env.f_obs(t, x)
            u = jnp.array([readout(y, a, 0, target)])
            return jnp.concatenate([env.diffusion(t, x, u), jnp.zeros((self.state_size, 2))])
        
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
        ys = jax.vmap(env.f_obs)(ts, xs)
        activities = sol.ys[:,self.latent_size:]
        us = jax.vmap(readout, in_axes=[0,0,None, None])(ys, activities, jnp.array([0]), target)
        fitness = env.fitness_function(xs, us, target)
        return xs, ys, us, activities, fitness
    
    #Selects a candidate from a randomly selected tournament. Selection is based on fitness and the probability of being chosen given a rank
    def tournament_selection(self, population, population_index):
        tournament = []
        #Sample tournament
        tournament_indices = jrandom.choice(self.keyGen.get(), self.population_size_array, shape=(self.tournament_size,), replace=False)
        for i in tournament_indices:
            tournament.append(population[i])
        #Sort on fitness
        tournament.sort(key=lambda x: x.fitness)
        index = jrandom.choice(self.keyGen.get(), self.tournament_indices, p=self.tournament_probabilities[population_index])
        return tournament[index]

    #Simplifies the tree if possible
    def simplify_algebra(self, tree):
        #Solve binary operator if the leaves does not contain variables
        if tree[0] in self.binary_operators:
            left_tree, simplify_left = self.simplify_algebra(tree[1])
            right_tree, simplify_right = self.simplify_algebra(tree[2])
           
            if simplify_left and simplify_right:
                return [self.binary_operators_map[tree[0]](left_tree[0], right_tree[0])], True
            tree = eqx.tree_at(lambda t: t[1], tree, left_tree)
            tree = eqx.tree_at(lambda t: t[2], tree, right_tree)
            if tree[0]=="/" or tree[0]=="*":
                if (left_tree[0]=="/" or left_tree[0]=="*") and simplify_right:
                    if isinstance(left_tree[1][0],jax.numpy.ndarray):
                        tree = eqx.tree_at(lambda t: t, tree, [left_tree[0],[self.binary_operators_map[tree[0]](left_tree[1][0], tree[2][0])],left_tree[2]])
                    elif isinstance(left_tree[2][0],jax.numpy.ndarray):
                        if left_tree[0]=="/" and tree[0]=="/":
                            tree = eqx.tree_at(lambda t: t, tree, [tree[0],left_tree[1],[self.binary_operators_map["*"](left_tree[2][0], tree[2][0])]])
                        elif left_tree[0]=="/":
                            tree = eqx.tree_at(lambda t: t, tree, [left_tree[0],left_tree[1],[self.binary_operators_map[left_tree[0]](left_tree[2][0], tree[2][0])]])
                        else:
                            tree = eqx.tree_at(lambda t: t, tree, [left_tree[0],left_tree[1],[self.binary_operators_map[tree[0]](left_tree[2][0], tree[2][0])]])
                        
                if (right_tree[0]=="/" or right_tree[0]=="*") and simplify_left:
                    if isinstance(right_tree[1][0],jax.numpy.ndarray):
                        if right_tree[0]=="/":
                            tree = eqx.tree_at(lambda t: t, tree, [right_tree[0],[self.binary_operators_map[tree[0]](tree[1][0],right_tree[1][0])],right_tree[2]])
                        else:
                            tree = eqx.tree_at(lambda t: t, tree, [tree[0],[self.binary_operators_map[tree[0]](tree[1][0],right_tree[1][0])],right_tree[2]])
                    elif isinstance(right_tree[2][0],jax.numpy.ndarray):
                        if right_tree[0]=="/":
                            tree = eqx.tree_at(lambda t: t, tree, [tree[0],[self.binary_operators_map[right_tree[0]](tree[1][0],right_tree[2][0])],right_tree[1]])
                        else:
                            tree = eqx.tree_at(lambda t: t, tree, [tree[0],[self.binary_operators_map[tree[0]](tree[1][0],right_tree[2][0])],right_tree[1]])
            if left_tree[0]==-1.0:
                if tree[0]=="+" and right_tree[0]=="*":
                    tree = ["-",tree[1],right_tree[2]]
            elif right_tree[0]==0:
                if tree[0]=="*":
                    tree = [jnp.array(0)]
                elif tree[0]=="**":
                    tree = [jnp.array(1)]
                else:
                    tree = left_tree
            elif left_tree[0]==0:
                if tree[0]=="*" or tree[0]=="/" or tree[0]=="**":
                    tree = [jnp.array(0)]
                elif tree[0]=="-":
                    tree = ["*",jnp.array(-1.0),tree[2]]
                else:
                    tree = right_tree
                
            return tree, False
        elif isinstance(tree[0],jax.numpy.ndarray):
            return tree, True #Possible to simplify
        else:
            return tree, False #Not possible to simplify
    
    def trees_to_sympy(self, trees):
        sympy_trees = []
        for tree in trees():
            sympy_trees.append(self.tree_to_sympy(tree))
        return sympy_trees

    def simplify_tree(self, tree):
        # print("Tree ", tree)
        expr = self.tree_to_sympy(tree)
        if expr == None:
            return None
        # print(expr)
        reconstructed_expr = self.reconstruct(expr, "Add" * isinstance(expr, sympy.Add) + "Mul" * isinstance(expr, sympy.Mul))
        # print(reconstructed_expr)
        try:
            new_tree = self.simplify_algebra(reconstructed_expr)[0]
        except:
            return None
        return new_tree
    
    def tree_to_sympy(self, tree):
        str_sol = self._to_string(tree)
        expr = sympy.parsing.sympy_parser.parse_expr(str_sol)
        if expr==sympy.nan or expr.has(sympy.core.numbers.ImaginaryUnit, sympy.core.numbers.ComplexInfinity):
            return None
        return expr

    def reconstruct(self, expr, mode):
        if isinstance(expr,sympy.Float) or isinstance(expr,sympy.Integer):
            return [jnp.array(float(expr))]
        elif isinstance(expr,sympy.Symbol):
            return [str(expr)]
        elif isinstance(expr,sympy.core.numbers.NegativeOne):
            return [jnp.array(-1)]
        elif isinstance(expr,sympy.core.numbers.Zero):
            return [jnp.array(0)]
        elif isinstance(expr,sympy.core.numbers.Half):
            return [jnp.array(0.5)]
        elif isinstance(expr*-1, sympy.core.numbers.Rational):
            return [jnp.array(float(expr))]
        elif not isinstance(expr,tuple):
            if isinstance(expr,sympy.Add):
                left_tree = self.reconstruct(expr.args[0], "Add")
                if len(expr.args)>2:
                    right_tree = self.reconstruct(expr.args[1:], "Add")
                else:
                    right_tree = self.reconstruct(expr.args[1], "Add")
                return ["+",left_tree,right_tree]
            if isinstance(expr,sympy.Mul):
                left_tree = self.reconstruct(expr.args[0], "Mul")
                if len(expr.args)>2:
                    right_tree = self.reconstruct(expr.args[1:], "Mul")
                else:
                    right_tree = self.reconstruct(expr.args[1], "Mul")
                return ["*",left_tree,right_tree]
            if isinstance(expr,sympy.cos):
                return ["cos",self.reconstruct(expr.args[0], mode=None)]
            if isinstance(expr,sympy.sin):
                return ["sin",self.reconstruct(expr.args[0], mode=None)]
            if isinstance(expr, sympy.Pow):
                if expr.args[1]==-1:
                    right_tree = self.reconstruct(expr.args[0], "Mul")
                    return ["/",[jnp.array(1)],right_tree]
                else:
                    left_tree = self.reconstruct(expr.args[0], "Add")
                    right_tree = self.reconstruct(expr.args[1], "Add")
                    return ["**", left_tree, right_tree]
        else:
            if mode=="Add":
                left_tree = self.reconstruct(expr[0], "Add")
                if len(expr)>2:
                    right_tree = self.reconstruct(expr[1:], "Add")
                else:
                    right_tree = self.reconstruct(expr[1], "Add")
                return ["+",left_tree,right_tree]
            if mode=="Mul":
                left_tree = self.reconstruct(expr[0], "Mul")
                if len(expr)>2:
                    right_tree = self.reconstruct(expr[1:], "Mul")
                else:
                    right_tree = self.reconstruct(expr[1], "Mul")
                return ["*",left_tree,right_tree]
        
    #Samples a random leaf. Either a contant or a variable
    def sample_leaf(self, sd=1):
        leaf_type = jrandom.uniform(self.keyGen.get(), shape=())
        if leaf_type<0.4:
            return [sd*jrandom.normal(self.keyGen.get())] #constant
        elif leaf_type<0.6: 
            return [self.variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.variables))]] #variable
        elif leaf_type<0.8:
            return [self.state_variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.state_variables))]]
        elif leaf_type<0.9:
            return [self.control_variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.control_variables))]]
        else:
            return ['target']
    
    #Generates a random node that can contain leaves at higher depths
    def grow_node(self, depth):
        #If depth is reached, a leaf is sampled
        if depth == 1:
            return self.sample_leaf(sd=3)
        leaf_type = jrandom.choice(self.keyGen.get(), jnp.arange(3), p=jnp.array([0.3,0.0,0.7])) #u=0.3
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
        
    #Generates a random node that can only have leaves at the deepest level
    def full_node(self, depth):
        #If depth is reached, a leaf is sampled
        if depth == 1:
            return self.sample_leaf(sd=3)
        leaf_type = jrandom.uniform(self.keyGen.get(), shape=())
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

    def sample_readout(self):
        return jrandom.normal(self.keyGen.get(), shape=(self.control_size, self.state_size))

    def sample_readout_leaf(self, sd=1):
        #Samples a random leaf. Either a contant or a variable
        leaf_type = jrandom.uniform(self.keyGen.get(), shape=())
        if leaf_type<0.4:
            return [sd*jrandom.normal(self.keyGen.get())] #constant
        elif leaf_type<0.9:
            return [self.state_variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.state_variables))]]
        else:
            return ['target']

    def sample_readout_tree(self, depth):
        if depth == 1:
            return self.sample_readout_leaf(sd=3)
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

    #Samples a population of trees with a certain (max) depth
    def sample_trees(self, max_depth = 3, N = 1, num_populations = 1, init_method="ramped"):
        assert (init_method=="ramped") or (init_method=="full") or (init_method=="grow"), "This method is not implemented"

        populations = []
        if init_method=="grow":
            for _ in range(num_populations):
                population = []
                while len(population) < N:
                    trees = []
                    for _ in range(self.state_size):
                        trees.append(self.grow_node(max_depth))
                    # readout = self.sample_readout()
                    readout = self.sample_readout_tree(max_depth)
                    new_individual = Expression(trees, readout)
                    if new_individual not in population:
                        population.append(new_individual)
                populations.append(population)
        elif init_method=="full":
            for _ in range(num_populations):
                population = []
                while len(population) < N:
                    trees = []
                    for _ in range(self.state_size):
                        trees.append(self.full_node(max_depth))
                    # readout = self.sample_readout()
                    readout = self.sample_readout_tree(max_depth)
                    new_individual = Expression(trees, readout)
                    if new_individual not in population:
                        population.append(new_individual)
                populations.append(population)
        elif init_method=="ramped":
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
                    # readout = self.sample_readout()
                    readout = self.sample_readout_tree(depth)
                    new_individual = Expression(trees, readout)
                    if new_individual not in population:
                        population.append(new_individual)
                populations.append(population)

        if N == 1:
            return populations[0][0]
        # if num_populations == 1:
        #     return populations[0]
        
        return populations

    #Return the subtree from a random node onwards
    def get_subtree(self, tree):
        leaves = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        distribution = jnp.array([0.5+1.5*self._operator(leaf) for leaf in leaves]) #increase selection probability of operators
        distribution = distribution.at[0].set(0.5) #lower probability of root
        index = jrandom.choice(self.keyGen.get(), jnp.arange(len(leaves)),p=distribution)

        path = flat_tree_and_path[index][0][:-1]

        subtree = self._key_loc(tree, path)

        return path, subtree
    
    #Performs standard cross-over on a pair of trees, returning two new trees. A cross-over point is selected in both trees, interchangin the subtrees behind this point
    def standard_cross_over(self, tree_a, tree_b):
        for i in range(self.state_size):
            path_a, subtree_a = self.get_subtree(tree_a()[i])
            path_b, subtree_b = self.get_subtree(tree_b()[i])

            tree_a = eqx.tree_at(lambda t: self._key_loc(t()[i], path_a), tree_a, subtree_b)
            tree_b = eqx.tree_at(lambda t: self._key_loc(t()[i], path_b), tree_b, subtree_a)

        path_a, subtree_a = self.get_subtree(tree_a.readout)
        path_b, subtree_b = self.get_subtree(tree_b.readout)

        tree_a = eqx.tree_at(lambda t: self._key_loc(t.readout, path_a), tree_a, subtree_b)
        tree_b = eqx.tree_at(lambda t: self._key_loc(t.readout, path_b), tree_b, subtree_a)

        return tree_a, tree_b
    
    def tree_cross_over(self, tree_a, tree_b):
        new_tree_a = tree_a
        new_tree_b = tree_b
        for i in range(self.state_size):
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: t()[i], new_tree_a, tree_b()[i])
                new_tree_b = eqx.tree_at(lambda t: t()[i], new_tree_b, tree_a()[i])
        if jrandom.uniform(self.keyGen.get()) > 0.5:
            new_tree_a = eqx.tree_at(lambda t: t.readout, new_tree_a, tree_b.readout)
            new_tree_b = eqx.tree_at(lambda t: t.readout, new_tree_b, tree_a.readout)
        return new_tree_a, new_tree_b

    #Performs uniform cross-over on a pair of trees, returning two new trees. Each overlapping node is switched with 50% chance and children of boundary nodes are switched as well.
    def uniform_cross_over(self, tree_a, tree_b):
        #Get intersection of the trees
        for i in range(self.state_size):
            interior_nodes, boundary_nodes = self._tree_intersection(tree_a()[i], tree_b()[i], path = [], interior_nodes = [], boundary_nodes = [])
            new_tree_a = tree_a
            new_tree_b = tree_b

            #Randomly switch two nodes of interior intersecting nodes
            for node in interior_nodes:
                if jrandom.uniform(self.keyGen.get()) > 0.5:
                    new_tree_a = eqx.tree_at(lambda t: self._index_loc(t()[i], node), new_tree_a, self._index_loc(tree_b()[i], node))
                    new_tree_b = eqx.tree_at(lambda t: self._index_loc(t()[i], node), new_tree_b, self._index_loc(tree_a()[i], node))
            #Randomly switch two nodes and their children of boundary intersecting nodes
            for node in boundary_nodes:
                if jrandom.uniform(self.keyGen.get()) > 0.5:
                    new_tree_a = eqx.tree_at(lambda t: self._index_loc(t()[i], node[:-1]), new_tree_a, self._index_loc(tree_b()[i], node[:-1]))
                    new_tree_b = eqx.tree_at(lambda t: self._index_loc(t()[i], node[:-1]), new_tree_b, self._index_loc(tree_a()[i], node[:-1]))
            
            tree_a = new_tree_a
            tree_b = new_tree_b

        interior_nodes, boundary_nodes = self._tree_intersection(tree_a.readout, tree_b.readout, path = [], interior_nodes = [], boundary_nodes = [])
        new_tree_a = tree_a
        new_tree_b = tree_b

        #Randomly switch two nodes of interior intersecting nodes
        for node in interior_nodes:
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: self._index_loc(t.readout, node), new_tree_a, self._index_loc(tree_b.readout, node))
                new_tree_b = eqx.tree_at(lambda t: self._index_loc(t.readout, node), new_tree_b, self._index_loc(tree_a.readout, node))
        #Randomly switch two nodes and their children of boundary intersecting nodes
        for node in boundary_nodes:
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: self._index_loc(t.readout, node[:-1]), new_tree_a, self._index_loc(tree_b.readout, node[:-1]))
                new_tree_b = eqx.tree_at(lambda t: self._index_loc(t.readout, node[:-1]), new_tree_b, self._index_loc(tree_a.readout, node[:-1]))
        
        tree_a = new_tree_a
        tree_b = new_tree_b

        return tree_a, tree_b

    #Determine the fitness of a tree. Different ways of determining the fitness are used in control and prediction tasks
    def get_fitness(self, tree, data, add_regularization=True):
        #Run coupled differential equations of state and control
        # try:
        x0, ts, targets, noise_keys, params = data
        _, _, _, _, fitness = jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0])(tree, x0, ts, targets, noise_keys, params)
        fitness = jnp.mean(fitness[:,-1])*self.dt
        
        if jnp.isinf(fitness) or jnp.isnan(fitness):
            fitness = jnp.array(1e6)
        if add_regularization:
            return jnp.clip(fitness + self.parsimony_punishment*len(jax.tree_util.tree_leaves(tree)),0,1e6)
        else:
            return jnp.clip(fitness,0,1e6)
    
    #Insert an operator at a random point in tree. Sample a new leaf if necessary to satisfy arity of the operator
    def insert_operator(self, tree, readout):
        nodes = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        operator_indices = jnp.ravel(jnp.argwhere(jnp.array([self._operator(node) for node in nodes])))
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
            if readout:
                other_leaf = self.sample_readout_leaf()
            else:
                other_leaf = self.sample_leaf()

            new_tree = [new_operator, subtree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, subtree]
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_tree)
    
    #Replace a leaf with a new subtree
    def add_subtree(self, tree, readout):
        nodes = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([self._leaf(node) for node in nodes])))
        index = jrandom.choice(self.keyGen.get(), leaf_indices)
        path = flat_tree_and_path[index][0][:-1]
        if readout:
            return eqx.tree_at(lambda t: self._key_loc(t, path), tree, self.sample_readout_tree(depth=3))
        else:
            return eqx.tree_at(lambda t: self._key_loc(t, path), tree, self.grow_node(depth=3))
    
    #Change an operator into a different operator of equal arity
    def mutate_operator(self, tree):
        nodes = jax.tree_util.tree_leaves(tree)
        operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._operator(node) for node in nodes])))
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

    #Add an operator to the top of the tree
    def prepend_operator(self, tree, readout):
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

    #Change value of a leaf. Leaf can stay the same type of change to the other leaf type
    def mutate_leaf(self, tree, readout):
        nodes = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([self._leaf(node) for node in nodes])))
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
    
    #Change the value of a constant leaf. The value is sampled around the old value
    def mutate_constant(self, tree):
        nodes = jax.tree_util.tree_leaves(tree)

        constant_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._constant(node) for node in nodes])))
        index = jrandom.choice(self.keyGen.get(), constant_indicies)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        value = flat_tree_and_path[index][1]
        path = flat_tree_and_path[index][0]
        #Sample around the old value
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, value+jrandom.normal(self.keyGen.get()))
    
    #Replace an operator with a new leaf
    def delete_operator(self, tree, readout):
        if readout:
            new_leaf = self.sample_readout_leaf(sd=3)
        else:
            new_leaf = self.sample_leaf(sd=3)
        nodes = jax.tree_util.tree_leaves(tree)
        operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._operator(node) for node in nodes])))
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree)
        index = jrandom.choice(self.keyGen.get(), operator_indicies)
        path = flat_tree_and_path[index][0][:-1]

        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_leaf)
    
    #Selects individuals that will replace randomly selected indivuals in a receiver distribution
    def migrate_trees(self, sender, receiver):
        #Select fitter individuals with higher probability
        sender_distribution = 1/jnp.array([p.fitness for p in sender])
        sender_indices = jrandom.choice(self.keyGen.get(), self.population_size_array, shape=(self.migration_size,), p=sender_distribution, replace=False)

        #Select unfit individuals with higher probability
        receiver_distribution = jnp.array([p.fitness for p in receiver])
        receiver_indices = jrandom.choice(self.keyGen.get(), self.population_size_array, shape=(self.migration_size,), p=receiver_distribution, replace=False)

        new_population = receiver

        for i in range(self.migration_size):
            new_population[receiver_indices[i]] = sender[sender_indices[i]]

        return new_population

    #Regulates the migration between pairs of populations
    def migrate_populations(self, populations):
        assert (self.migration_method=="ring") or (self.migration_method=="random"), "This method is not implemented"
        if self.num_populations==1:
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

    def mutate_tree(self, tree, allow_simplification=True, readout=False):
        mod_prob = self.mod_prob.copy()

        if len(tree)==1:
            mod_prob = mod_prob.at[self.mutations.index("mutate_operator")].set(0)
            mod_prob = mod_prob.at[self.mutations.index("delete_operator")].set(0)
            mod_prob = mod_prob.at[self.mutations.index("insert_operator")].set(0)
        if sum([self._constant(node) for node in jax.tree_util.tree_leaves(tree)]) == 0:
            mod_prob = mod_prob.at[self.mutations.index("mutate_constant")].set(0)
        if not allow_simplification:
            mod_prob = mod_prob.at[self.mutations.index("simplify_tree")].set(0)

        mutation_type = self.mutations[jrandom.choice(self.keyGen.get(), jnp.arange(len(self.mutations)), shape=(), p=mod_prob)]
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
                new_tree = self.sample_readout_tree(depth=3)
            else:
                new_tree = self.grow_node(depth=3)
        elif mutation_type=="add_subtree":
            new_tree = self.add_subtree(tree, readout)
        elif mutation_type=="simplify_tree":
            old_tree_string = sympy.parsing.sympy_parser.parse_expr(self._to_string(tree),evaluate=False)
            new_tree_string = sympy.parsing.sympy_parser.parse_expr(self._to_string(tree))
            if old_tree_string==new_tree_string:
                new_tree = self.mutate_tree(tree, allow_simplification=False, readout=readout)
            
            simplified_tree = self.simplify_tree(tree)
            if simplified_tree==None:
                new_tree = self.mutate_tree(tree, allow_simplification=False, readout=readout)
            else:
                new_tree = simplified_tree
        if (self._tree_depth(new_tree) > self.max_depth):
            new_tree = self.mutate_tree(tree,readout=readout)
        return new_tree

    def similarity(self, tree_a, tree_b):
        #beide leaves
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
        return int(tree_a[0] == tree_b[0]) + max(self.similarity(tree_a[1], tree_b[1]) + self.similarity(tree_a[2], tree_b[2]), self.similarity(tree_a[1], tree_b[2]) + self.similarity(tree_a[2], tree_b[1]))

    def similarities(self, tree_a, tree_b):
        similarity = 0
        for i in range(self.state_size):
            similarity += self.similarity(tree_a()[i],tree_b()[i])/min(len(jax.tree_util.tree_leaves(tree_a()[i])),len(jax.tree_util.tree_leaves(tree_b()[i])))
        return similarity/self.state_size

    def mutate_readout(self, readout):
        return self.mutate_tree(readout, readout=True)

    def next_population(self, population, mean_fitness, population_index):
        remaining_candidates = len(population)
        #Keep best candidate in new population
        new_pop = []
        failed_mutations = 0

        #Loop until new population has reached the desired size
        while remaining_candidates>1:
            probs = self.probabilities[population_index]
            tree_a = self.tournament_selection(population, population_index)
            tree_b = self.tournament_selection(population, population_index)

            similarity = self.similarities(tree_a, tree_b)
            if tree_a.fitness > mean_fitness and tree_b.fitness > mean_fitness:
                probs = [0,0.5,0.5,0]

            elif similarity > self.similarity_threshold:
                probs = [0,0.6,0.3,0.1]
            action = jrandom.choice(self.keyGen.get(), jnp.arange(4), p=jnp.array(probs))
                
            if action==0:
                cross_over_type = jrandom.uniform(self.keyGen.get())              
                if cross_over_type < 0.3:
                    new_tree_a, new_tree_b = self.tree_cross_over(tree_a, tree_b)
                elif cross_over_type < 0.7:
                    new_tree_a, new_tree_b = self.uniform_cross_over(tree_a, tree_b)
                else:
                    new_tree_a, new_tree_b = self.standard_cross_over(tree_a, tree_b)

                #If both trees remain the same, cross-over has failed
                if eqx.tree_equal(tree_a, new_tree_a) and eqx.tree_equal(tree_b, new_tree_b):
                    failed_mutations += 1
                elif (self._tree_depth(new_tree_a) > self.max_depth) or (self._tree_depth(new_tree_b) > self.max_depth):
                    failed_mutations += 1
                else:
                    #Append new trees to the new generation
                    new_pop.append(new_tree_a)
                    new_pop.append(new_tree_b)
                    remaining_candidates -= 2

            elif action==1:
                mutate_bool = jrandom.uniform(self.keyGen.get(), shape=(self.state_size,))
                while not any(mutate_bool>self.tree_mutate_probs[population_index]):
                    mutate_bool = jrandom.uniform(self.keyGen.get(), shape=(self.state_size,))
                new_tree_a = tree_a
                for i in range(self.state_size):
                    if mutate_bool[i]>self.tree_mutate_probs[population_index]:
                        new_tree = self.mutate_tree(tree_a()[i])
                        new_tree_a = eqx.tree_at(lambda t: t()[i], new_tree_a, new_tree)
                if jrandom.uniform(self.keyGen.get()) > self.tree_mutate_probs[population_index]:
                    new_tree_a = eqx.tree_at(lambda t: t.readout, new_tree_a, self.mutate_readout(new_tree_a.readout))
                new_pop.append(new_tree_a)
                remaining_candidates -= 1

                mutate_bool = jrandom.uniform(self.keyGen.get(), shape=(self.state_size,))
                while not any(mutate_bool>self.tree_mutate_probs[population_index]):
                    mutate_bool = jrandom.uniform(self.keyGen.get(), shape=(self.state_size,))
                new_tree_b = tree_b
                for i in range(self.state_size):
                    if mutate_bool[i]>self.tree_mutate_probs[population_index]:
                        new_tree = self.mutate_tree(tree_b()[i])
                        new_tree_b = eqx.tree_at(lambda t: t()[i], new_tree_b, new_tree)
                if jrandom.uniform(self.keyGen.get()) > self.tree_mutate_probs[population_index]:
                    new_tree_b = eqx.tree_at(lambda t: t.readout, new_tree_b, self.mutate_readout(new_tree_b.readout))
                new_pop.append(new_tree_b)
                remaining_candidates -= 1

            elif action==2:
                new_trees = self.sample_trees(max_depth=3, N=2, init_method="full")[0]
                #Add new tree to the new generation
                remaining_candidates -= 2
                new_pop.append(new_trees[0])
                new_pop.append(new_trees[1])
            elif action==3:
                remaining_candidates -= 2
                new_pop.append(tree_a)
                new_pop.append(tree_b)
        
        best_candidate = sorted(population, key=lambda x: x.fitness, reverse=False)[0]
        if remaining_candidates==0:
            new_pop[0] = best_candidate
        else:
            new_pop.append(best_candidate)
        return new_pop

    #Returns the type of the root of a tree
    def get_root(self, tree):
        root = tree()[0]
        if isinstance(root, jax.numpy.ndarray):
            root = "Constant"
        return root
    
    #Plots the number of nodes, depth, root type and fitness of all individuals to show the diversity of the populations
    def plot_statistics(self, population):
        fig, ax = plt.subplots(2, 2, figsize=(15,10))
        ax[0,0].hist(jnp.log(jnp.array([tree.fitness for tree in population])), align="left")
        ax[0,0].set_title("Fitness of each tree in the final population")
        ax[0,0].set_xlabel("Log of fitness")
        ax[0,0].set_ylabel("Frequency")

        ax[0,1].hist([self.get_root(tree) for tree in population], align="left")
        ax[0,1].set_title("Root type of each tree in the final population")
        ax[0,1].set_xlabel("Root type")
        ax[0,1].set_ylabel("Frequency")

        ax[1,0].hist([self._tree_depth(tree) for tree in population], align="left")
        ax[1,0].set_title("Tree depth of each tree in the final population")
        ax[1,0].set_xlabel("Depth")
        ax[1,0].set_ylabel("Frequency")

        ax[1,1].hist([len(jax.tree_util.tree_leaves(tree)) for tree in population],align="left")
        ax[1,1].set_title("Number of nodes in each tree in the final population")
        ax[1,1].set_xlabel("Number of nodes")
        ax[1,1].set_ylabel("Frequency")

        plt.show()
    
    #Finds the best solution in all populations
    def _best_solution(self, populations, fitnesses):
        best_solution = None
        best_fitness = jnp.inf
        for pop in range(self.num_populations):
            # print(jnp.min(fitnesses[pop]))
            if jnp.min(fitnesses[pop]) < best_fitness:
                best_fitness = jnp.min(fitnesses[pop])
                best_solution = populations[pop][jnp.argmin(fitnesses[pop])]
        return best_solution
    
    def flatten(self, populations):
        return [candidate for population in populations for candidate in population]

    def initialize_variables(self, env):
        #Define parameters that may be included in solutions
        self.env = env
        self.obs_size = env.n_obs
        self.latent_size = env.n_var
        self.control_size = env.n_control
        self.variables = ["y" + str(i) for i in range(self.obs_size)]
        self.state_variables = ["a" + str(i) for i in range(self.state_size)]
        self.control_variables = ["u" + str(i) for i in range(self.control_size)]
        self.dt = env.dt

    def random_search(self, env, data, num_generations, pool_size, plot_statistics = False, continue_population=None, converge_value=0):
        pool = Pool(pool_size)
        pool.close()

        self.initialize_variables(env)

        best_fitnesses = jnp.zeros(num_generations)
        best_solutions = []
        if continue_population is None:
            #Initialise new population
            populations = self.sample_trees(self.max_depth, self.population_size, num_populations=self.num_populations, init_method=self.init_method)
        else:
            #Continue from previous population
            populations = continue_population
        
        for g in range(num_generations):
            pool.restart()
            fitness = pool.amap(lambda x: self.get_fitness(x,data),self.flatten(populations))
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
            # fitnesses = []
            for pop in range(self.num_populations):
                population = populations[pop]
                for candidate in range(self.population_size):
                    population[candidate].set_fitness(fitnesses[pop,candidate])
            best_fitnesses = best_fitnesses.at[g].set(self.get_fitness(self._best_solution(populations, fitnesses), data, add_regularization=False))

            if best_fitnesses[g]<converge_value:
                best_solution = self._best_solution(populations, fitnesses)
                try:
                    best_solution_string = self.trees_to_sympy(best_solution)
                except:
                    print("Error in best, ", best_solution)
                best_solutions.append(best_solution)
                print(f"Converge settings satisfied, best fitness {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout)}")

                best_fitnesses = best_fitnesses.at[g:].set(best_fitnesses[g])

                if plot_statistics:
                    self.plot_statistics(self.flatten(populations))

                break

            elif g < num_generations-1:
                best_solution = self._best_solution(populations, fitnesses)
                try:
                    best_solution_string = self.trees_to_sympy(best_solution)
                except:
                    print("Error in best, ", best_solution)
                print(f"In generation {g+1}, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout)}")
                              
                best_candidates = []
                for pop in range(self.num_populations):
                    best_candidates.append(sorted(populations[pop], key=lambda x: x.fitness, reverse=False)[0])
                populations = self.sample_trees(self.max_depth, self.population_size-1, num_populations=self.num_populations, init_method=self.init_method)
                for pop in range(self.num_populations):
                    populations[pop].append(best_candidates[pop])
                
            else:
                best_solution = self._best_solution(populations, fitnesses)
                try:
                    best_solution_string = self.trees_to_sympy(best_solution)
                except:
                    print("Error in best, ", best_solution)
                best_solutions.append(best_solution)
                print(f"Final generation, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout)}")
            
                if plot_statistics:
                    self.plot_statistics(self._flatten_populations(populations))

        return best_fitnesses, best_solutions, populations

    #Runs the GP algorithm for a given number of runs and generations. It is possible to continue on a previous population or start from a new population.
    def run(self, env, data, num_generations, pool_size, plot_statistics = False, continue_population=None, converge_value=0, insert_solution=None):
        pool = Pool(pool_size)
        pool.close()

        self.initialize_variables(env)

        best_fitnesses = jnp.zeros(num_generations)
        best_fitness_per_population = jnp.zeros((num_generations, self.num_populations))
        best_solutions = []
        if continue_population is None:
            #Initialise new population
            populations = self.sample_trees(self.max_init_depth, self.population_size, num_populations=self.num_populations, init_method=self.init_method)
        else:
            #Continue from previous population
            populations = continue_population

        if insert_solution is not None:
            populations[0][0] = insert_solution
        
        for g in range(num_generations):
            pool.restart()
            fitness = pool.amap(lambda x: self.get_fitness(x,data),self.flatten(populations))
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
            for pop in range(self.num_populations):
                population = populations[pop]
                for candidate in range(self.population_size):
                    population[candidate].set_fitness(fitnesses[pop,candidate])
                populations[pop] = population

            best_fitnesses = best_fitnesses.at[g].set(self.get_fitness(self._best_solution(populations, fitnesses), data, add_regularization=False))

            if (best_fitnesses[g]==best_fitnesses[g-5]) and best_fitnesses[g]<converge_value:
                best_solution = self._best_solution(populations, fitnesses)
                try:
                    best_solution_string = self.trees_to_sympy(best_solution)
                except:
                    print("Error in best, ", best_solution)
                best_solutions.append(best_solution)
                print(f"Converge settings satisfied, best fitness {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout)}")

                best_fitnesses = best_fitnesses.at[g:].set(best_fitnesses[g])

                if plot_statistics:
                    self.plot_statistics(self.flatten(populations))

                break

            elif g < num_generations-1:
                best_solution = self._best_solution(populations, fitnesses)
                try:
                    best_solution_string = self.trees_to_sympy(best_solution)
                except:
                    print("Error in best, ", best_solution)
                print(f"In generation {g+1}, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout)}")
                
                #Migrate individuals between populations every few generations
                if ((g+1)%self.migration_period)==0:
                    populations = self.migrate_populations(populations)
                
                for pop in range(self.num_populations):
                    #Perform operations to generate new population
                    if best_fitness_per_population[g,pop] == best_fitness_per_population[g-self.restart_iter,pop]:
                        print(f"Restart in population {pop+1} at generation {g+1}")
                        populations[pop] = self.sample_trees(self.max_init_depth, self.population_size, num_populations=1, init_method=self.init_method)[0]
                    else:
                        populations[pop] = self.next_population(populations[pop], jnp.mean(flat_fitnesses), pop)
            else:
                best_solution = self._best_solution(populations, fitnesses)
                try:
                    best_solution_string = self.trees_to_sympy(best_solution)
                except:
                    print("Error in best, ", best_solution)
                best_solutions.append(best_solution)
                print(f"Final generation, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution_string}, readout: {self.tree_to_sympy(best_solution.readout)}")
            
                if plot_statistics:
                    self.plot_statistics(self._flatten_populations(populations))

        return best_fitnesses, best_solutions, populations