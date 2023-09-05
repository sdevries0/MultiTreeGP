import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
from jax.tree_util import register_pytree_node
import diffrax
import treelib
import matplotlib.pyplot as plt
import time

class KeyGen:
    def __init__(self, seed = 0):
        self._key = jrandom.PRNGKey(seed)

    def get(self) -> jrandom.PRNGKeyArray:
        self._key, new_key = jrandom.split(self._key)
        return new_key

#Container that contains a symbolic expression. Might be redundant, but used to clearly store expressions
class Expression:
    def __init__(self, expression):
        self.expression = expression
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
        return f"{[self._to_string(tree) for tree in self.expression]}"
    
#Register the container as pytree
register_pytree_node(Expression, lambda tree: ((tree.expression,), None),
    lambda _, args: Expression(*args))

class ODE_GP:
    def __init__(self, environment, fitness_function, state_size, control_size, tournament_size = 4, selection_pressure = 0.9, max_depth=10, max_init_depth=5, population_size=100, num_populations=1, migration_period=4, migration_percentage=0.1, migration_method="ring", init_method="ramped"):
        self.keyGen = KeyGen(0)
        self.environment = environment
        self.fitness_function = fitness_function

        #Define operators
        self.binary_operators_map = {"+":lambda a,b:a+b, "-":lambda a,b:a-b, "*":lambda a,b:jnp.clip(a*b, a_min=-100, a_max=100), "/":lambda a,b:jnp.clip(a/(b+1e-5), a_min=-100, a_max=100)}
        self.unary_operators_map = {"sin":lambda a:jnp.sin(a), "cos":lambda a:jnp.cos(a)}#,"exp":lambda a:jnp.clip(jnp.exp(a),a_max=100), "sqrt":lambda a:jnp.sqrt(jnp.abs(a))}
        self.binary_operators = list(self.binary_operators_map.keys())
        self.unary_operators = list(self.unary_operators_map.keys())

        #Define modification types
        self.modifications = ["reproduction","cross_over","mutate_operator","delete_operator","insert_operator","mutate_constant","mutate_leaf","sample_tree","sample_subtree","prepend_operator","add_subtree","simplify_tree"]
        self.mod_prob = jnp.ones(len(self.modifications))*0
        self.mod_prob = self.mod_prob.at[self.modifications.index("cross_over")].set(6.5)
        self.mod_prob = self.mod_prob.at[self.modifications.index("reproduction")].set(1)
        self.mod_prob = self.mod_prob.at[self.modifications.index("sample_tree")].set(1)
        self.mod_prob = self.mod_prob.at[self.modifications.index("sample_subtree")].set(1)
        self.mod_prob = self.mod_prob.at[self.modifications.index("simplify_tree")].set(0.5)
        # self.mod_prob = self.mod_prob.at[self.modifications.index("mutate_operator")].set(0.47)
        # self.mod_prob = self.mod_prob.at[self.modifications.index("mutate_constant")].set(0.048)
        # self.mod_prob = self.mod_prob.at[self.modifications.index("delete_operator")].set(1.7)
        # self.mod_prob = self.mod_prob.at[self.modifications.index("insert_operator")].set(5.1)
        # self.mod_prob = self.mod_prob.at[self.modifications.index("mutate_leaf")].set(0.1)
        # self.mod_prob = self.mod_prob.at[self.modifications.index("prepend_operator")].set(0.4)
        # self.mod_prob = self.mod_prob.at[self.modifications.index("add_subtree")].set(0.4)
        
        #Define parameters that may be included in solutions
        self.obs_size = environment.n_obs
        self.state_size = state_size
        self.variables = ["x" + str(i) for i in range(self.obs_size)]
        self.state_variables = ["a" + str(i) for i in range(self.state_size)]
        self.control_size = control_size
        
        #Specify GP settings
        self.tournament_size = tournament_size
        self.tournament_probabilities = jnp.array([selection_pressure*(1-selection_pressure)**n for n in jnp.arange(self.tournament_size)])
        self.tournament_indices = jnp.arange(self.tournament_size)
        self.temperature = 1
        self.temperature_decay = 0.9
        self.parsimony_punishment = 0.1
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.population_size = population_size
        self.population_size_array = jnp.arange(population_size)
        self.num_populations = num_populations
        self.migration_period = migration_period
        self.migration_size = int(migration_percentage*self.population_size)
        self.migration_method = migration_method
        self.init_method = init_method

    #Returns subtree at location specified by a path
    def _key_loc(self, tree, path):
        new_tree = tree
        for k in path:
            new_tree = new_tree[k.idx]
        return new_tree
    
    #Returns subtree at location specified by indices
    def _index_loc(self, tree, path):
        new_tree = tree()
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
    
    #Fill externally defined parameters into a tree
    def fill_params(self, tree, params, paths):
        for param, path in zip(params, paths):
            tree = eqx.tree_at(lambda t: self._key_loc(t, path), tree, param)
        return tree

    #Fills in values for variables
    def fill_variables(self, tree, x, a, target, t=None):
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree())
        for key_path, leaf in flat_tree_and_path:
            #If t is included in trees, replace its value with current timepoint
            if leaf == "target":
                tree = eqx.tree_at(lambda t: self._key_loc(t(), key_path), tree, target)
            elif leaf in self.state_variables:
                tree = eqx.tree_at(lambda t: self._key_loc(t(), key_path), tree, a[self.state_variables.index(leaf)])
            elif leaf in self.variables:
                tree = eqx.tree_at(lambda t: self._key_loc(t(), key_path), tree, x[self.variables.index(leaf)])
                
        return tree
    
    #Computes the outcome of the expression. Assumses that variables are given values
    def evaluate_tree(self, tree):
        if tree[0] in self.binary_operators:
            assert len(tree) == 3, f"The operator {tree[0]} requires two inputs"
            return self.binary_operators_map[tree[0]](self.evaluate_tree(tree[1]),self.evaluate_tree(tree[2]))
        
        elif tree[0] in self.unary_operators:
            assert len(tree) == 2, f"The operator {tree[0]} requires one input"
            return self.unary_operators_map[tree[0]](self.evaluate_tree(tree[1]))
    
        assert len(tree) == 1, "Leaves should not have children"
        return tree[0]
    
    def evaluate_expression(self, expression):
        return jnp.array([self.evaluate_tree(tree) for tree in expression()])
    
    #Solves the coupled differential equation of the state and the control. The differential equation of the state is defined in the environment and the differential equation of the control is defined by a tree.
    def evaluate_control_loop(self, tree, x0, ts, target, key):
        #Define state equation
        def _drift(t, x_a, args):
            x = x_a[:self.obs_size]
            a = x_a[self.obs_size:]
            u = a[:self.control_size]
            #Get environment update
            dx, y = self.environment.drift(t, x, u)
            sol_tree = self.fill_variables(tree, y, a, target)
            da = self.evaluate_expression(sol_tree)
            return jnp.concatenate([dx, da])
        
        #Define diffusion
        def _diffusion(t, x_a, args):
            x = x_a[:self.obs_size]
            a = x_a[self.obs_size:]
            u = a[:self.control_size]
            # return self.environment.diffusion(t, x, args)
            return jnp.concatenate([self.environment.diffusion(t, x, u), jnp.zeros((self.state_size, 2))])
        
        solver = diffrax.Tsit5()
        dt0 = 0.1
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        brownian_motion = diffrax.VirtualBrownianTree(ts[0], ts[-1], tol=1e-1, shape=(2,), key=key)
        system = diffrax.MultiTerm(diffrax.ODETerm(_drift), diffrax.ControlTerm(_diffusion, brownian_motion))    
        # system = diffrax.ODETerm(_drift)    

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat#, stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6)
        )
        ys = sol.ys[:,:self.obs_size]
        activities = sol.ys[:,self.obs_size:]
        us = activities[:,:self.control_size]
        return ys, us, activities
    
    #Selects a candidate from a randomly selected tournament. Selection is based on fitness and the probability of being chosen given a rank
    def tournament_selection(self, population):
        tournament = []
        #Sample tournament
        tournament_indices = jrandom.choice(self.keyGen.get(), self.population_size_array, shape=(self.tournament_size,), replace=False)
        for i in tournament_indices:
            tournament.append(population[i])
        #Sort on fitness
        tournament.sort(key=lambda x: x.fitness, reverse=False)
        index = jrandom.choice(self.keyGen.get(), self.tournament_indices, p=self.tournament_probabilities)
        return tournament[index]

    #Simplifies the tree if possible
    def simplify_tree(self, tree):
        #Solve binary operator if the leaves does not contain variables
        if tree[0] in self.binary_operators:
            left_tree, simplify_left = self.simplify_tree(tree[1])
            right_tree, simplify_right = self.simplify_tree(tree[2])
            #Replace a/a with 1
            if tree[0] == "/":
                if eqx.tree_equal(left_tree, right_tree):
                    return [1.0], True
            if simplify_left and simplify_right:
                return [self.binary_operators_map[tree[0]](left_tree[0], right_tree[0])], True
            elif simplify_left:
                return eqx.tree_at(lambda t: t[1], tree, left_tree), False
            elif simplify_right:
                return eqx.tree_at(lambda t: t[2], tree, right_tree), False
            else:
                return tree, False
        #Solve unary operator if the leaf does not contain a variable
        elif tree[0] in self.unary_operators:
            left_tree, simplify_left = self.simplify_tree(tree[1])
            if simplify_left:
                return [self.unary_operators_map[tree[0]](left_tree[0])], True
            else:
                #Replace log(exp(a)) with a (or other way around)
                if (tree[0] == "exp" and tree[1][0] == "log") or (tree[0] == "log" and tree[1][0] == "exp"):
                    return tree[1][1], False
                return tree, False
        elif isinstance(tree[0],jax.numpy.ndarray):
            return tree, True #Possible to simplify
        else:
            return tree, False #Not possible to simplify
        
    def simplify_trees(self, trees):
        for i in range(self.state_size):
            trees = eqx.tree_at(lambda t: t()[i], trees, self.simplify_tree(trees()[i])[0])
        return trees
        
    #Samples a random leaf. Either a contant or a variable
    def sample_leaf(self, sd=1):
        leaf_type = jrandom.uniform(self.keyGen.get(), shape=())
        if leaf_type<0.4:
            return [sd*jrandom.normal(self.keyGen.get())] #constant
        elif leaf_type<0.6: 
            return [self.variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.variables))]] #variable
        elif leaf_type<0.9:
            return [self.state_variables[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.state_variables))]]
        else:
            return ['target']
    
    #Generates a random node that can contain leaves at higher depths
    def grow_node(self, depth):
        #If depth is reached, a leaf is sampled
        if depth == 1:
            return self.sample_leaf(sd=3)
        leaf_type = jrandom.choice(self.keyGen.get(), jnp.arange(3), p=jnp.array([0.3,0.2,0.5])) #u=0.3
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
        if leaf_type > 0.75: #unary operator
            tree.append(self.unary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.unary_operators))])
            tree.append(self.full_node(depth-1))
            return tree
        else: #binary operator
            tree.append(self.binary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.binary_operators))])
            tree.append(self.full_node(depth-1))
            tree.append(self.full_node(depth-1))
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
                    new_individual = Expression(trees)
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
                    new_individual = Expression(trees)
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
                    new_individual = Expression(trees)
                    if new_individual not in population:
                        population.append(new_individual)
                populations.append(population)

        if N == 1:
            return populations[0][0]
        
        return populations
        
    def sample_subtree(self, trees, max_depth = 3):
        tree_index = jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=self.state_size)
        return eqx.tree_at(lambda t: t()[tree_index], trees, self.full_node(max_depth))

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

        return tree_a, tree_b
    
    def tree_cross_over(self, tree_a, tree_b):
        new_tree_a = tree_a
        new_tree_b = tree_b
        for i in range(self.state_size):
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: t()[i], new_tree_a, tree_b()[i])
                new_tree_b = eqx.tree_at(lambda t: t()[i], new_tree_b, tree_a()[i])
        return new_tree_a, new_tree_b

    #Performs uniform cross-over on a pair of trees, returning two new trees. Each overlapping node is switched with 50% chance and children of boundary nodes are switched as well.
    def uniform_cross_over(self, tree_a, tree_b):
        #Get intersection of the trees
        interior_nodes, boundary_nodes = self._tree_intersection(tree_a(), tree_b(), path = [], interior_nodes = [], boundary_nodes = [])
        new_tree_a = tree_a
        new_tree_b = tree_b

        #Randomly switch two nodes of interior intersecting nodes
        for i in interior_nodes:
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: self._index_loc(t, i), new_tree_a, self._index_loc(tree_b, i))
                new_tree_b = eqx.tree_at(lambda t: self._index_loc(t, i), new_tree_b, self._index_loc(tree_a, i))
        #Randomly switch two nodes and their children of boundary intersecting nodes
        for b in boundary_nodes:
            if jrandom.uniform(self.keyGen.get()) > 0.5:
                new_tree_a = eqx.tree_at(lambda t: self._index_loc(t, b[:-1]), new_tree_a, self._index_loc(tree_b, b[:-1]))
                new_tree_b = eqx.tree_at(lambda t: self._index_loc(t, b[:-1]), new_tree_b, self._index_loc(tree_a, b[:-1]))

        return new_tree_a, new_tree_b

    #Determine the fitness of a tree. Different ways of determining the fitness are used in control and prediction tasks
    def get_fitness(self, tree, y0, ts, targets, keys, add_regularization=True):
        #Run coupled differential equations of state and control
        states, control, _ = jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0])(tree, y0, ts, targets, keys)
        if jnp.sum(jnp.isinf(states)) > 0 or jnp.sum(jnp.isnan(states)) > 0:
            fitness = jnp.array(1e5)
        else:
            #Compute reward, turn it into fitness and add complixty punishment
            fitness = -1*jnp.mean(jax.vmap(self.fitness_function)(states, control, targets)[:,-1])*self.environment.dt 
            if jnp.isinf(fitness) or jnp.isnan(fitness):
                fitness = jnp.array(1e5)
        
        if add_regularization:
            return jnp.clip(fitness + self.parsimony_punishment*len(jax.tree_util.tree_leaves(tree)),0,1e5)
        else:
            return jnp.clip(fitness,0,1e5)
    
    #Insert an operator at a random point in tree. Sample a new leaf if necessary to satisfy arity of the operator
    def insert_operator(self, tree):
        if len(tree())==1:
            return tree, True
        leaves = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree())
        operator_indices = jnp.ravel(jnp.argwhere(jnp.array([self._operator(leaf) for leaf in leaves])))
        index = jrandom.choice(self.keyGen.get(), operator_indices)
        path = flat_tree_and_path[index][0][:-1]
        subtree = self._key_loc(tree, path)
        
        operator_type = jrandom.uniform(self.keyGen.get())
        if operator_type>0.6: #unary operator
            new_operator = self.unary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.unary_operators))]
            new_tree = [new_operator, subtree]
        else: #binary operator
            new_operator = self.binary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.binary_operators))]
            tree_position = jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=2)
            other_leaf = self.sample_leaf()

            new_tree = [new_operator, subtree, other_leaf] if (tree_position == 0) else [new_operator, other_leaf, subtree]
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_tree), False
    
    #Replace a leaf with a new subtree
    def add_subtree(self, tree):
        leaves = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree())
        leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([self._leaf(leaf) for leaf in leaves])))
        index = jrandom.choice(self.keyGen.get(), leaf_indices)
        path = flat_tree_and_path[index][0][:-1]
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, self.full_node(depth=2))
    
    #Change an operator into a different operator of equal arity
    def mutate_operator(self, trees):
        tree_index = jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=self.state_size)
        tree = trees()[tree_index]
        #If tree does not contain operators, return as failed mutation
        if len(tree)==1:
            return tree, True
        leaves = jax.tree_util.tree_leaves(tree)
        operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._operator(leaf) for leaf in leaves])))
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
        return eqx.tree_at(lambda t: t()[tree_index], trees, new_tree), False

    #Add an operator to the top of the tree
    def prepend_operator(self, tree):
        if jrandom.uniform(self.keyGen.get())>0.6:
            new_operator = self.unary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.unary_operators))]
            new_tree = [new_operator, tree()]
        else:
            new_operator = self.binary_operators[jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=len(self.binary_operators))]
            tree_position = jrandom.randint(self.keyGen.get(), shape=(), minval=0, maxval=2)
            #Sample a leaf for the other child of the operator
            other_leaf = self.sample_leaf()

            new_tree = [new_operator, tree(), other_leaf] if (tree_position == 0) else [new_operator, other_leaf, tree()]
        return eqx.tree_at(lambda t: t(), tree, new_tree)

    #Change value of a leaf. Leaf can stay the same type of change to the other leaf type
    def mutate_leaf(self, tree):
        leaves = jax.tree_util.tree_leaves(tree)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree())
        leaf_indices = jnp.ravel(jnp.argwhere(jnp.array([self._leaf(leaf) for leaf in leaves])))
        index = jrandom.choice(self.keyGen.get(), leaf_indices)
        path = flat_tree_and_path[index][0][:-1]
        new_leaf = self.sample_leaf(sd=3)
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_leaf)
    
    #Change the value of a constant leaf. The value is sampled around the old value
    def mutate_constant(self, tree):
        leaves = jax.tree_util.tree_leaves(tree)
        if sum([self._constant(leaf) for leaf in leaves]) == 0:
            return tree, True
        constant_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._constant(leaf) for leaf in leaves])))
        index = jrandom.choice(self.keyGen.get(), constant_indicies)
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree())
        value = flat_tree_and_path[index][1]
        path = flat_tree_and_path[index][0]
        #Sample around the old value, with a temperature controlling the variance
        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, value+self.temperature*jrandom.normal(self.keyGen.get())), False
    
    #Replace an operator with a new leaf
    def delete_operator(self, tree):
        #If tree does not contain operators, return as failed mutation
        if len(tree())==1:
            return tree, True
            
        new_leaf = self.sample_leaf(sd=3)
        leaves = jax.tree_util.tree_leaves(tree)
        operator_indicies = jnp.ravel(jnp.argwhere(jnp.array([self._operator(leaf) for leaf in leaves])))
        flat_tree_and_path = jax.tree_util.tree_leaves_with_path(tree())
        index = jrandom.choice(self.keyGen.get(), operator_indicies)
        path = flat_tree_and_path[index][0][:-1]

        return eqx.tree_at(lambda t: self._key_loc(t, path), tree, new_leaf), False
    
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

    #Generate new trees from the old generation. Trees with better fitness have a bigger chance to reproduce
    def next_population(self, population):
        remaining_candidates = len(population)-1
        #Keep best candidate in new population
        best_candidate = sorted(population, key=lambda x: x.fitness, reverse=False)[0]
        new_pop = [best_candidate]
        failed_mutations = 0

        #Loop until new population has reached the desired size
        while remaining_candidates>0:
            #If only one spot is left, do not choose cross-over
            if remaining_candidates > 1:
                #Sample modification to be applied on a tree
                modification_type = self.modifications[jrandom.choice(self.keyGen.get(), jnp.arange(len(self.modifications)), shape=(), p=self.mod_prob)]
            else:
                #Sample modification to be applied on a tree, but crossover is not allowed
                mod_prob_copy = self.mod_prob.at[self.modifications.index("cross_over")].set(0)
                modification_type = self.modifications[jrandom.choice(self.keyGen.get(), jnp.arange(len(self.modifications)), shape=(), p=mod_prob_copy)]
    
            if modification_type=="cross_over":
                tree_a = self.tournament_selection(population)
                tree_b = self.tournament_selection(population)

                #only apply uniform cross-over when there is more than one point of intersection
                # if len(tree_a()) == len(tree_b()):
                #     if jrandom.uniform(self.keyGen.get()) > 0.7:
                #         new_tree_a, new_tree_b = self.standard_cross_over(tree_a, tree_b)
                #     else:
                #         new_tree_a, new_tree_b = self.uniform_cross_over(tree_a, tree_b)
                # else:
                #     new_tree_a, new_tree_b = self.standard_cross_over(tree_a, tree_b)

                if jrandom.uniform(self.keyGen.get()) > 0.5:
                    new_tree_a, new_tree_b = self.tree_cross_over(tree_a, tree_b)
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
            else:
                #Mutate a single tree to produce one new tree. Some mutations can fail
                trees = self.tournament_selection(population)
                failed = False
                if modification_type=="reproduction":
                    new_trees = trees   
                elif modification_type=="mutate_operator":
                    new_trees, failed = self.mutate_operator(trees)
                elif modification_type=="delete_operator":
                    new_trees, failed = self.delete_operator(trees)
                elif modification_type=="prepend_operator":
                    new_trees = self.prepend_operator(trees)
                elif modification_type=="insert_operator":
                    new_trees, failed = self.insert_operator(trees)
                elif modification_type=="mutate_constant":
                    new_trees, failed = self.mutate_constant(trees)
                elif modification_type=="mutate_leaf":
                    new_trees = self.mutate_leaf(trees)
                elif modification_type=="sample_tree":
                    new_trees = self.sample_trees(init_method="full")
                elif modification_type=="sample_subtree":
                    new_trees = self.sample_subtree(trees)
                elif modification_type=="add_subtree":
                    new_trees = self.add_subtree(trees)
                elif modification_type=="simplify_tree":
                    simplified_tree, _ = self.simplify_tree(trees())
                    new_trees = eqx.tree_at(lambda t: t(), trees, simplified_tree)
                    if eqx.tree_equal(trees, new_trees):
                        failed = True
                if (self._tree_depth(new_trees) > self.max_depth) or failed:
                    failed_mutations += 1
                else:
                    #Add new tree to the new generation
                    remaining_candidates -= 1
                    new_pop.append(new_trees)
        
        #Decrease the temperature used for controlling the variance of mutating new constants
        self.temperature = self.temperature * self.temperature_decay
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
    
    #Visualize part of a tree
    def visualize_sub_tree(self, tree, c=0):
        v_tree = treelib.Tree()
        if len(tree) == 1:
            v_tree.create_node(str(tree[0]), f"{c}")
            new_c = c
        else:
            v_tree.create_node(tree[0], f"{c}")
            if len(tree) == 3:
                left_tree, new_c = self.visualize_sub_tree(tree[1], c+1)
                v_tree.paste(f"{c}",left_tree)
                right_tree, new_c = self.visualize_sub_tree(tree[2], new_c+1)
                v_tree.paste(f"{c}",right_tree)
            if len(tree) == 2:
                left_tree, new_c = self.visualize_sub_tree(tree[1], c+1)
                v_tree.paste(f"{c}",left_tree)
        return v_tree, new_c
        
    #Visualize tree as a graph
    def visualize_tree(self, tree, simplify=False):
        if simplify:
            tree, _ = self.simplify_tree(tree())
        v_tree, _ = self.visualize_sub_tree(tree())
        v_tree.show()

    #Flattens populations
    def _flatten_populations(self, populations):
        return [p for pop in populations for p in pop]

    #Finds the best solution in all populations
    def _best_solution(self, populations, fitnesses):
        best_solution = None
        best_fitness = jnp.inf
        for pop in range(self.num_populations):
            if jnp.min(fitnesses[pop]) < best_fitness:
                best_fitness = jnp.min(fitnesses[pop])
                best_solution = populations[pop][jnp.argmin(fitnesses[pop])]
        return best_solution

    #Runs the GP algorithm for a given number of runs and generations. It is possible to continue on a previous population or start from a new population.
    def run(self, data, n_trials, num_generations, plot_statistics = False, continue_population=None, converge_value=0):
        best_fitnesses = jnp.zeros((n_trials, num_generations))
        best_solutions = []
        y0, ts, targets = data
        for r in range(n_trials):
            keys = jrandom.split(self.keyGen.get(), y0.shape[0])
            if continue_population is None:
                #Initialise new population
                populations = self.sample_trees(self.max_init_depth, self.population_size, num_populations=self.num_populations, init_method=self.init_method)
            else:
                #Continue from previous population
                populations = continue_population
            
            for g in range(num_generations):
                fitnesses = jnp.zeros((self.num_populations, self.population_size))
                for pop in range(self.num_populations):
                    population = populations[pop]
                    for candidate in range(self.population_size):
                        #Get fitness of a candidate
                        fitness = self.get_fitness(population[candidate], y0, ts, targets, keys)
                        population[candidate].set_fitness(fitness)
                        fitnesses = fitnesses.at[pop, candidate].set(fitness)
                #Get fitness of best solution of all populations without regularisation
                best_fitnesses = best_fitnesses.at[r, g].set(self.get_fitness(self._best_solution(populations, fitnesses), y0, ts, targets, keys, add_regularization=False))

                if (best_fitnesses[r,g]==best_fitnesses[r,g-2]) and best_fitnesses[r,g]<converge_value:
                    best_solution = self._best_solution(populations, fitnesses)
                    best_solutions.append(best_solution)
                    print(f"Converge settings satisfied, best fitness {jnp.min(fitnesses)}, best solution: {best_solution}")

                    best_fitnesses = best_fitnesses.at[r, g:].set(best_fitnesses[r,g])

                    if plot_statistics:
                        self.plot_statistics(self._flatten_populations(populations))

                    break

                elif g < num_generations-1:
                    print(f"In generation {g}, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}")
                    
                    #Migrate individuals between populations every few generations
                    if ((g+1)%self.migration_period)==0:
                        populations = self.migrate_populations(populations)
                    
                    for pop in range(self.num_populations):
                        #Perform operations to generate new population
                        populations[pop] = self.next_population(populations[pop])
                else:
                    best_solution = self._best_solution(populations, fitnesses)
                    best_solutions.append(best_solution)
                    print(f"Final generation, average fitness: {jnp.mean(fitnesses)}, best_fitness: {jnp.min(fitnesses)}, best solution: {best_solution}")
                
                    if plot_statistics:
                        self.plot_statistics(self._flatten_populations(populations))

        return best_fitnesses, best_solutions, populations