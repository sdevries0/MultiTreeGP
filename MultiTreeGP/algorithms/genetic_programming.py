import warnings

import jax
print(jax.devices())
from jax import Array

import jax.numpy as jnp
import jax.random as jr
from jax.random import PRNGKey
import optax
import equinox as eqx
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import sympy

from typing import Tuple, Callable
import copy
import time

from MultiTreeGP.networks.tree_policy import TreePolicy
from MultiTreeGP.expression import OperatorNode
import MultiTreeGP.genetic_operators.reproduction as reproduction
import MultiTreeGP.genetic_operators.migration as migration
import MultiTreeGP.genetic_operators.initialization as initialization
from MultiTreeGP.algorithms.strategy_base import Strategy

def find_best_solution(populations: list, fitnesses: Array) -> TreePolicy:
    """Returns the best solution in all subpopulations.

        :param populations: Population of candidates
        :param fitnesses: The fitness of all candidates.

        Returns: Best candidate.
        """
    best_solution = None
    best_fitness = jnp.inf
    for pop in range(len(populations)):
        if jnp.min(fitnesses[pop]) < best_fitness:
            best_fitness = jnp.min(fitnesses[pop])
            best_solution = populations[pop][jnp.argmin(fitnesses[pop])]
    return best_solution

def lambda_func1(f):
    return lambda x, y, _data: f(x)

def lambda_func2(f):
    return lambda x, y, _data: f(x, y)

def lambda_leaf(i):
    return lambda x, y, _data: _data[i]

class GeneticProgramming(Strategy):
    """Genetic programming strategy of symbolic expressions.

        Attributes:
            num_generations: The number of generations over which to evolve the population.
            population_size: Number of candidates in the population.
            fitness_function: Function that evaluates a candidate and assigns a fitness.
            current_generation: A reference to keep track of the current generation.
            best_solutions: Best solution at each generation.
            best_fitnesses: Best fitness at each generation.
            expressions: Expressions for each layer in a tree policy.
            layer_sizes: Size of each layer in a tree policy.
            num_populations: Number of subpopulations.
            max_depth: Highest depth of a tree.
            max_init_depth: Highest depth of a tree during initialization.
            init_method: Method for initializing the trees.
            size_parsinomy: Parsimony factor that increases the fitness of a candidate based on its size.
            leaf_sd: Standard deviation to sample constants.
            migration_period: Number of generations after which migration happens.
            migration_method: Method that specifies how migration takes place.
            migration_size: Number of candidates to migrate.
            restart_iter_threshold: Number of generations that a subpopulation did not improve before a restart occurs.
            last_restart: Counter that keeps track of the last restart of each subpopulation.
            tournament_size: Size of the tournament.
            selection_pressures: Selection pressure in a tournament, varying per population.
            tournament_probabilities: Probability of each rank in a tournament to be selected for reproduction.
            reproduction_type_probabilities: Probability of each reproduction method to be selected, varying per population.
            reproduction_probabilities: Probability of a tree to be adapted in a candidate, varying per population.
            elite_percentage: Percentage of elite candidates that procceed to the next population, varying per population.
            mutation_probabilities: Probability for each mutation method to be selected.
            best_fitness_per_population: The best fitness in each subpopulation.
        """
    def __init__(self, num_generations: int, 
                 population_size: int, 
                 fitness_function: Callable, 
                 expressions: list, 
                 layer_sizes: Array, 
                 num_populations: int = 1, 
                 max_depth: int = 7, 
                 max_init_depth: int = 4, 
                 max_nodes: int = 40,
                 tournament_size: int = 7, 
                 init_method: str = "ramped", 
                 size_parsinomy: float = 1.0, 
                 migration_period: int = 5,
                 leaf_sd: float = 1.0, 
                 migration_percentage: float = 0.1,
                 restart_iter_threshold: int = None,
                 gradient_optimisation: bool = False,
                 gradient_steps: int = 10) -> None:
        # assert len(layer_sizes) == len(expressions), "There is not a set of expressions for every type of layer"
        self.expressions = expressions
        self.layer_sizes = layer_sizes
        self.num_populations = num_populations
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.max_nodes = min(max_nodes, 2**self.max_depth-1)
        self.num_trees = jnp.sum(self.layer_sizes)
        super().__init__(num_generations, population_size, fitness_function, (self.num_trees, self.max_nodes, 4))

        self.init_method = init_method
        if (init_method=="ramped") or (init_method=="full"):
            assert 2**self.max_init_depth < self.max_nodes, "Full trees are not possible given initialization depth and max size"

        self.size_parsinomy = size_parsinomy
        self.leaf_sd = leaf_sd

        self.migration_period = migration_period
        self.migration_size = int(migration_percentage*population_size)

        self.gradient_optimisation = gradient_optimisation
        self.gradient_steps = gradient_steps

        if restart_iter_threshold == None:
            with warnings.catch_warnings(action="ignore"):
                self.restart_iter_threshold: int = jnp.linspace(10, 5, num_populations, dtype=int)
        else:
            self.restart_iter_threshold = restart_iter_threshold
        self.last_restart = jnp.zeros(num_populations)

        self.tournament_size = tournament_size
        self.selection_pressures = jnp.linspace(0.6,0.9,self.num_populations)
        self.tournament_probabilities = jnp.array([sp*(1-sp)**jnp.arange(self.tournament_size) for sp in self.selection_pressures])
        
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(0.9,0.2,self.num_populations),jnp.linspace(0.1,0.8,self.num_populations)
                                                           ,jnp.linspace(0.0,0.0,self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(1.0, 0.5, self.num_populations)
        self.elite_percentage = jnp.linspace(0.04, 0.04)

        self.mutate_functions = [self.add_subtree, self.mutate_leaf, self.mutate_operator, self.delete_operator, self.prepend_operator, self.insert_operator, self.replace_tree]

        self.best_fitness_per_population = jnp.zeros((num_generations, self.num_populations))

        self.optimizer = optax.adam(learning_rate=0.01, b1=0.8, b2=0.9)

        func_dict = {}
        func_to_string = {}
        functions = [lambda x, y, _data: 0.0, lambda x, y, _data: 0.0]

        self.map_b_to_d = self.create_map_b_to_d(self.max_depth)

        self.jit_evolve_population = jax.jit(self.evolve_population)

        index = -2
        data_index = 0
        slots = [0, 0]

        self.func_i = -2

        for expression in self.expressions:
            for operator in expression.operators:
                if operator.string not in func_dict:
                    func_dict[operator.string] = index
                    func_to_string[index] = operator.string
                    if operator.arity==1:
                        functions.append(lambda_func1(operator.f))
                        slots.append(1)
                    elif operator.arity==2:
                        functions.append(lambda_func2(operator.f))
                        slots.append(2)
                    index -= 1
        self.func_j = index + 1

        for expression in self.expressions:
            for leaf in expression.leaf_nodes:
                if leaf.string not in func_dict:
                    func_dict[leaf.string] = index
                    func_to_string[index] = leaf.string
                    functions.append(lambda_leaf(data_index))
                    slots.append(0)
                    index -= 1
                    data_index += 1
        
        self.leaf_j = index + 1

        self.slots = jnp.array(slots)

        self.func_dict = func_dict
        self.func_to_string = func_to_string
        self.functions = functions

        self.partial_ff = partial(self.fitness_function, eval = self.vmap_foriloop)
        self.jit_optimise = jax.vmap(self.optimise, in_axes=[0, None, None])
        self.vmap_trees = jax.jit(jax.vmap(self.partial_ff, in_axes=[0, None]))
        self.jit_body_fun = jax.jit(partial(self.body_fun, functions = self.functions))

    def create_map_b_to_d(self, depth):
        max_nodes = 2**depth-1
        current_depth = 0
        map_b_to_d = jnp.zeros(max_nodes)
        for i in range(max_nodes):
            if i>0:
                parent = (i + (i%2) - 2)//2
                value = map_b_to_d[parent]
                if (i % 2)==0:
                    new_value = value + 2**(depth-current_depth+1)
                else:
                    new_value = value + 1
                map_b_to_d = map_b_to_d.at[i].set(new_value)
            current_depth += i==(2**current_depth-1)
        return max_nodes - 1 - map_b_to_d

    def sample_FS_node(self, i, carry):
        key, matrix, open_slots, max_depth = carry
        float_key, leaf_key, variable_key, node_key, func_key = jr.split(key, 5)
        _i = self.map_b_to_d[i].astype(int)

        depth = (jnp.log(i+1)/jnp.log(2)).astype(int)
        float = jr.normal(float_key)*self.leaf_sd
        leaf = jax.lax.select(jr.uniform(leaf_key)<0.5, -1, jr.randint(variable_key, shape=(), minval=self.leaf_j, maxval=self.func_j))
        index = jax.lax.select((open_slots < self.max_nodes - i - 1) & (depth<max_depth), 
                               jax.lax.select(jr.uniform(node_key)<(0.7**depth), 
                                              jr.choice(func_key, a=jnp.arange(self.func_j, self.func_i+1), shape=(), p=jnp.flip(self.expressions[0].operators_prob)), 
                                              leaf), 
                               leaf)
        index = jax.lax.select(open_slots == 0, 0, index)
        index = jax.lax.select(i>0, jax.lax.select((self.slots[-jnp.minimum(matrix[self.map_b_to_d[(i + (i%2) - 2)//2].astype(int), 0], 0).astype(int)] + i%2) > 1, index, 0), index)

        matrix = jax.lax.select(self.slots[-1*index] > 0, matrix.at[_i, 1].set(self.map_b_to_d[2*i+1]), matrix.at[_i, 1].set(-1))
        matrix = jax.lax.select(self.slots[-1*index] > 1, matrix.at[_i, 2].set(self.map_b_to_d[2*i+2]), matrix.at[_i, 2].set(-1))

        matrix = jax.lax.select(index == -1, matrix.at[_i,3].set(float), matrix)
        matrix = matrix.at[_i, 0].set(index)

        open_slots = jax.lax.select(index == 0, open_slots, jnp.maximum(0, open_slots + self.slots[-1*index] - 1))

        return (jr.fold_in(key, i), matrix, open_slots, max_depth)
    
    def prune_row(self, i, carry, old_matrix):
        matrix, counter = carry

        _i = 2**self.max_depth - i - 2

        row = old_matrix[_i]

        matrix = jax.lax.select(row[0] != 0, matrix.at[counter].set(row), matrix.at[:,1:3].set(jnp.where(matrix[:,1:3] > _i, matrix[:,1:3]-1, matrix[:,1:3])))
        counter = jax.lax.select(row[0] != 0, counter - 1, counter)

        return (matrix, counter)
        
    def prune_tree(self, matrix):
        matrix, counter = jax.lax.fori_loop(0, 2**self.max_depth-1, partial(self.prune_row, old_matrix=matrix), (jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1)), self.max_nodes-1))
        matrix = matrix.at[:,1:3].set(jnp.where(matrix[:,1:3]>-1, matrix[:,1:3] + counter + 1, matrix[:,1:3]))
        return matrix

    def sample_tree(self, key, depth):
        tree = jax.lax.fori_loop(0, 2**self.max_depth-1, self.sample_FS_node, (key, jnp.zeros((2**self.max_depth-1, 4)), 1, depth))[1]
        return self.prune_tree(tree)
    
    def sample_trees(self, keys, depth):
        return jax.vmap(self.sample_tree, in_axes=[0, None])(keys, depth)
    
    def sample_population(self, key):
        return jax.vmap(self.sample_trees, in_axes=[0, None])(jr.split(key, (self.population_size, self.num_trees)), self.max_init_depth)
        
    def initialize_population(self, key: PRNGKey) -> list:
        """Randomly initializes the population.

        :param key: Random key

        Returns: Population.
        """
        keys = jr.split(key, self.num_populations)
        populations = jax.vmap(self.sample_population)(keys)

        return populations
    
    def tree_to_string(self, tree):
        if tree[-1,0]==-1:
            return "{:.2f}".format(tree[-1,3])
        elif tree[-1,1]<0:
            return self.func_to_string[tree[-1,0].astype(int).item()]
        elif tree[-1,2]<0:
            substring = self.tree_to_string(tree[:tree[-1,1].astype(int)+1])
            return f"{self.func_to_string[tree[-1,0].astype(int).item()]}({substring})"
        else:
            substring1 = self.tree_to_string(tree[:tree[-1,1].astype(int)+1])
            substring2 = self.tree_to_string(tree[:tree[-1,2].astype(int)+1])
            return f"({substring1}){self.func_to_string[tree[-1,0].astype(int).item()]}({substring2})"
        
    def to_string(self, trees):
        string_output = ""
        tree_index = 0
        layer_index = 0
        for tree in trees:
            if tree_index==0:
                string_output += "["
            string_output += str(sympy.parsing.sympy_parser.parse_expr(self.tree_to_string(tree)))
            if tree_index < (self.layer_sizes[layer_index] - 1):
                string_output += ", "
                tree_index += 1
            else:
                string_output += "]"
                if layer_index < (self.layer_sizes.shape[0] - 1):
                    string_output += ", "
                tree_index = 0
                layer_index += 1
        return string_output
    
    def body_fun(self, i, carry, functions):
        array, data = carry
        f_idx, a_idx, b_idx, float = array[i]
    
        x = array[a_idx.astype(int), 3]
        y = array[b_idx.astype(int), 3]
        value = jax.lax.select(f_idx == -1, float, jax.lax.switch(-f_idx.astype(int), functions, x, y, data))
        
        array = array.at[i, 3].set(value)

        return (array, data)

    def foriloop(self, data, array):
        x, _ = jax.lax.fori_loop(0, self.max_nodes, self.jit_body_fun, (array, data))
        return data, x[-1, -1]

    def vmap_foriloop(self, array, data):
        _, result = jax.lax.scan(self.foriloop, init=data, xs=array)
        # result = jax.vmap(self.foriloop, in_axes=[0, None, None])(array, data, self.max_nodes)
        return result
    
    def evaluate_population(self, populations: list, data: Tuple) -> Tuple[Array, list]:
        """Evaluates every candidate in population and assigns a fitness.

        :param population: Population of candidates
        :param data: The data required to evaluate the population.

        Returns: Fitness and evaluated population.
        """
        devices = mesh_utils.create_device_mesh((len(jax.devices('cpu'))))
        mesh = Mesh(devices, axis_names=('i'))
        
        @partial(shard_map, mesh=mesh, in_specs=(P('i'), P(None)), out_specs=(P('i'), P('i')), check_rep=False)
        def shard_eval(array, data):
            result = self.vmap_trees(array, data)
            return result, array
        
        @partial(shard_map, mesh=mesh, in_specs=(P('i'), P(None)), out_specs=(P('i'), P('i')), check_rep=False)
        def shard_optimise(array, data):
            result, _array = self.jit_optimise(array, data, self.gradient_steps)
            return result, _array
        
        flat_populations = populations.reshape(self.num_populations*self.population_size, *populations.shape[2:])
        flat_populations = jax.device_put(flat_populations, NamedSharding(mesh, P('i')))
        
        fitness, optimised_population = shard_optimise(flat_populations, data) if self.gradient_optimisation else shard_eval(flat_populations, data)
        fitness = fitness + jax.vmap(lambda array: self.size_parsinomy * jnp.sum(array[:,:,0]!=0))(flat_populations)
        # fitness = self.vmap_trees(flat_populations, data)
        # fitness, optimised_population = shard_optimise(flat_populations, data)
        # print(gradients)
        fitness = fitness.reshape((self.num_populations, self.population_size))

        self.best_fitness_per_population = self.best_fitness_per_population.at[self.current_generation].set(jnp.min(fitness, axis=1))
        best_solution_of_g = optimised_population[jnp.argmin(fitness)]
        best_fitness_of_g = jnp.min(fitness)

        #Keep track of best solution
        if self.current_generation == 0:
            best_fitness = best_fitness_of_g
            best_solution = best_solution_of_g
        else:
            best_fitness = self.best_fitnesses[self.current_generation - 1]
            best_solution = self.best_solutions[self.current_generation - 1]

            if best_fitness_of_g < best_fitness:
                best_fitness = best_fitness_of_g
                best_solution = best_solution_of_g
            
        self.best_solutions = self.best_solutions.at[self.current_generation].set(best_solution)
        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)

        return fitness, optimised_population.reshape((self.num_populations, self.population_size, *optimised_population.shape[1:]))      
            
    def epoch(self, i, carry):
        tree, state, data, _ = carry
        # loss, gradient = jax.value_and_grad(self.partial_ff)(jnp.concatenate([static_tree, _tree[:,:,None]], axis=2), data)
        loss, gradient = jax.value_and_grad(self.partial_ff)(tree, data)

        # print(gradient.shape, _tree.shape)
        # new_tree = tree - 0.1 * gradient
        updates, state = self.optimizer.update(gradient, state, tree)
        new_tree = tree + updates
        # return (tree, state), loss
        return new_tree, state, data, loss

    def optimise(self, tree, data: Tuple, n_steps):
        """Evaluates a candidate and assigns a fitness and optionally optimises the constants in the tree.

        :param candidate: Candidate solution.
        :param data: The data required to evaluate the population.
        :param optimise: Whether to optimise the constants in the tree.
        :param num_steps: Number of steps during constant optimisation.

        Returns: Optimised and evaluated candidate.
        """
        fitness = self.partial_ff(tree, data)
        
        state = self.optimizer.init(tree)

        # (final_tree, _), loss = jax.lax.scan(epoch, (tree, state), xs=[], length=1)
        # _tree, state, loss = jax.lax.fori_loop(0, 1, epoch, (tree[:,:,3], state, 0))
        new_tree, _, _, loss = jax.lax.fori_loop(0, n_steps, self.epoch, (tree, state, data, 0))

        fitness = jax.lax.select(loss < fitness, loss, fitness)
        tree = jax.lax.select(loss < fitness, new_tree, tree)

        return fitness, tree
    
    def sample_indices(self, carry):
        _key, prev, reproduction_probability = carry
        indices = jr.bernoulli(_key, p=reproduction_probability, shape=prev.shape)*1.0
        return (jr.split(_key, 1)[0], indices, reproduction_probability)

    def find_end_idx(self, carry):
        tree, openslots, counter = carry
        _, idx1, idx2, _ = tree[counter]
        openslots -= 1
        openslots = jax.lax.select(idx1 < 0, openslots, openslots+1)
        openslots = jax.lax.select(idx2 < 0, openslots, openslots+1)
        counter -= 1
        return (tree, openslots, counter)
    
    def check_invalid_cx_points(self, carry):
        tree1, tree2, _, node_idx1, node_idx2 = carry

        _, _, end_idx1 = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree1, 1, node_idx1))
        _, _, end_idx2 = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree2, 1, node_idx2))

        subtree_size1 = node_idx1 - end_idx1
        subtree_size2 = node_idx2 - end_idx2

        empty_nodes1 = jnp.sum(tree1[:,0]==0)
        empty_nodes2 = jnp.sum(tree2[:,0]==0)

        return (empty_nodes1 < subtree_size2 - subtree_size1) | (empty_nodes2 < subtree_size1 - subtree_size2)

    def sample_cx_points(self, carry):
        tree1, tree2, keys, _, _ = carry
        key1, key2 = keys

        cx_prob1 = (tree1[:,0] <= self.func_i) & (tree1[:,0] >= self.func_j)
        cx_prob1 = jnp.where(tree1[:,0]==0, cx_prob1, cx_prob1+1)
        node_idx1 = jr.choice(key1, jnp.arange(self.max_nodes), p = cx_prob1*1.)

        cx_prob2 = (tree2[:,0] <= self.func_i) & (tree2[:,0] >= self.func_j)
        cx_prob2 = jnp.where(tree2[:,0]==0, cx_prob2, cx_prob2+1)
        node_idx2 = jr.choice(key2, jnp.arange(self.max_nodes), p = cx_prob2*1.)

        return (tree1, tree2, jr.split(key1), node_idx1, node_idx2)
    
    def crossover(self, tree1, tree2, keys):
        #Define indices of the nodes
        tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        key1, key2 = keys

        #Define last node in tree
        last_node_idx1 = jnp.sum(tree1[:,0]==0)
        last_node_idx2 = jnp.sum(tree2[:,0]==0)

        #Randomly select nodes for crossover
        cx_prob1 = (tree1[:,0] <= self.func_i) & (tree1[:,0] >= self.func_j)
        cx_prob1 = jnp.where(tree1[:,0]==0, cx_prob1, cx_prob1+1)
        node_idx1 = jr.choice(key1, jnp.arange(self.max_nodes), p = cx_prob1*1.)

        cx_prob2 = (tree2[:,0] <= self.func_i) & (tree2[:,0] >= self.func_j)
        cx_prob2 = jnp.where(tree2[:,0]==0, cx_prob2, cx_prob2+1)
        node_idx2 = jr.choice(key2, jnp.arange(self.max_nodes), p = cx_prob2*1.)

        #Reselect until valid crossover points have been found
        _, _, _, node_idx1, node_idx2 = jax.lax.while_loop(self.check_invalid_cx_points, self.sample_cx_points, (tree1, tree2, jr.split(key1), node_idx1, node_idx2))

        #Retrieve subtrees of selected nodes
        _, _, end_idx1 = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree1, 1, node_idx1))
        _, _, end_idx2 = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree2, 1, node_idx2))

        #Initialize children
        child1 = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1))
        child2 = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1))

        #Compute subtree sizes
        subtree_size1 = node_idx1 - end_idx1
        subtree_size2 = node_idx2 - end_idx2

        #Insert nodes before subtree in children
        child1 = jnp.where(tree_indices >= node_idx1 + 1, tree1, child1)
        child2 = jnp.where(tree_indices >= node_idx2 + 1, tree2, child2)
        
        #Align nodes after subtree with first open spot after new subtree in children
        rolled_tree1 = jnp.roll(tree1, subtree_size1 - subtree_size2, axis=0)
        rolled_tree2 = jnp.roll(tree2, subtree_size2 - subtree_size1, axis=0)

        #Insert nodes after subtree in children
        child1 = jnp.where((tree_indices >= node_idx1 - subtree_size2 - (end_idx1 - last_node_idx1)) & (tree_indices < node_idx1 + 1 - subtree_size2), rolled_tree1, child1)
        child2 = jnp.where((tree_indices >= node_idx2 - subtree_size1 - (end_idx2 - last_node_idx2)) & (tree_indices < node_idx2 + 1 - subtree_size1), rolled_tree2, child2)

        #Update index references to moved nodes in staying nodes
        child1 = child1.at[:,1:3].set(jnp.where((child1[:,1:3] < (node_idx1 - subtree_size1 + 1)) & (child1[:,1:3] > -1), child1[:,1:3] + (subtree_size1-subtree_size2), child1[:,1:3]))
        child2 = child2.at[:,1:3].set(jnp.where((child2[:,1:3] < (node_idx2 - subtree_size2 + 1)) & (child2[:,1:3] > -1), child2[:,1:3] + (subtree_size2-subtree_size1), child2[:,1:3]))

        #Align subtree with the selected node in children
        rolled_subtree1 = jnp.roll(tree1, node_idx2 - node_idx1, axis=0)
        rolled_subtree2 = jnp.roll(tree2, node_idx1 - node_idx2, axis=0)

        #Update index references in subtree
        rolled_subtree1 = rolled_subtree1.at[:,1:3].set(jnp.where(rolled_subtree1[:,1:3] > -1, rolled_subtree1[:,1:3] + (node_idx2 - node_idx1), -1))
        rolled_subtree2 = rolled_subtree2.at[:,1:3].set(jnp.where(rolled_subtree2[:,1:3] > -1, rolled_subtree2[:,1:3] + (node_idx1 - node_idx2), -1))

        #Insert subtree in selected node in children
        child1 = jnp.where((tree_indices >= node_idx1 + 1 - subtree_size2) & (tree_indices < node_idx1 + 1), rolled_subtree2, child1)
        child2 = jnp.where((tree_indices >= node_idx2 + 1 - subtree_size1) & (tree_indices < node_idx2 + 1), rolled_subtree1, child2)
        
        return child1, child2
    
    def crossover_trees(self, parent1, parent2, keys, reproduction_probability):
        _, cx_indices, _ = jax.lax.while_loop(lambda carry: jnp.sum(carry[1])==0, self.sample_indices, (keys[0, 0], jnp.zeros(parent1.shape[0]), reproduction_probability))
        offspring1, offspring2 = jax.vmap(self.crossover)(parent1, parent2, keys)
        child1 = jnp.where(cx_indices[:,None,None] * jnp.ones_like(parent1), offspring1, parent1)
        child2 = jnp.where(cx_indices[:,None,None] * jnp.ones_like(parent2), offspring2, parent2)
        return child1, child2

    def add_subtree(self, tree, key):
        tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        select_key, sample_key = jr.split(key, 2)
        node_ids = tree[:,0]
        is_leaf = (node_ids == -1) | (node_ids < self.func_j)
        mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.)

        subtree = self.sample_tree(sample_key, depth=2)
        subtree_size = jnp.sum(subtree[:,0]!=0)

        remaining_size = mutate_idx - jnp.sum(tree[:,0]==0)

        child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1))
        child = jnp.where(tree_indices > mutate_idx, tree, child)

        rolled_tree = jnp.roll(tree, -subtree_size + 1, axis=0)
        child = jnp.where((tree_indices <= mutate_idx - subtree_size) & (tree_indices > mutate_idx - subtree_size - remaining_size), rolled_tree, child)
        child = child.at[:,1:3].set(jnp.where((child[:,1:3] < (mutate_idx)) & (child[:,1:3] > -1), child[:,1:3] - (subtree_size - 1), child[:,1:3]))

        subtree = jnp.roll(subtree, -(self.max_nodes - mutate_idx - 1), axis=0)
        subtree = subtree.at[:,1:3].set(jnp.where(subtree[:,1:3] > -1, subtree[:,1:3] + (mutate_idx - self.max_nodes + 1), -1))
        child = jnp.where((tree_indices <= mutate_idx) & (tree_indices > mutate_idx - subtree_size), subtree, child)

        return child

    def sample_leaf_point(self, carry):
        tree, key, _, _ = carry
        key, select_key, sample_key, variable_key = jr.split(key, 4)
        node_ids = tree[:,0]
        is_leaf = (node_ids == -1) | (node_ids < self.func_j)
        mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.)
        new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, -1, jr.randint(variable_key, shape=(), minval=self.leaf_j, maxval=self.func_j))

        return (tree, key, mutate_idx, new_leaf)
    
    def check_equal_leaves(self, carry):
        tree, key, mutate_idx, new_leaf = carry

        return (tree[mutate_idx, 0] == new_leaf) & (new_leaf != -1)

    def mutate_leaf(self, tree, key):
        select_key, sample_key, float_key, variable_key = jr.split(key, 4)
        node_ids = tree[:,0]
        is_leaf = (node_ids == -1) | (node_ids < self.func_j)
        mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_leaf*1.)
        new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, -1, jr.randint(variable_key, shape=(), minval=self.leaf_j, maxval=self.func_j))
        _, _, mutate_idx, new_leaf = jax.lax.while_loop(self.check_equal_leaves, self.sample_leaf_point, (tree, jr.fold_in(key, 0), mutate_idx, new_leaf))

        float = jr.normal(float_key)

        child = tree.at[mutate_idx, 0].set(new_leaf)
        child = jax.lax.select(new_leaf==-1, child.at[mutate_idx, 3].set(float), child.at[mutate_idx, 3].set(0))

        return child

    def replace_with_one_subtree(self, tree, mutate_idx, operator, key):
        tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree, 1, mutate_idx))

        remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1

        subtree = self.sample_tree(key, depth=2)
        subtree_size = jnp.sum(subtree[:,0]!=0)

        child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1))

        child = jnp.where(tree_indices >= mutate_idx, tree, child)

        rolled_tree = jnp.roll(tree, (mutate_idx - end_idx - subtree_size - 1), axis=0)
        child = jnp.where((tree_indices < mutate_idx - subtree_size) & (tree_indices >= mutate_idx - subtree_size - remaining_size), rolled_tree, child)

        child = child.at[mutate_idx, 0].set(operator)
        child = child.at[mutate_idx, 2].set(-1)

        child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] + (mutate_idx - end_idx - subtree_size - 1), child[:,1:3]))

        subtree = jnp.roll(subtree, -(self.max_nodes - mutate_idx), axis=0)
        subtree = subtree.at[:,1:3].set(jnp.where(subtree[:,1:3] > -1, subtree[:,1:3] + (mutate_idx - self.max_nodes), -1))
        child = jnp.where((tree_indices < mutate_idx) & (tree_indices > mutate_idx - subtree_size - 1), subtree, child)

        return child

    def replace_with_two_subtrees(self, tree, mutate_idx, operator, key):
        tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        key1, key2 = jr.split(key)
        _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree, 1, mutate_idx))

        remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1

        subtree1 = self.sample_tree(key1, depth=1)
        subtree1_size = jnp.sum(subtree1[:,0]!=0)
        subtree2 = self.sample_tree(key2, depth=1)
        subtree2_size = jnp.sum(subtree2[:,0]!=0)

        child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1))
        child = jnp.where(tree_indices >= mutate_idx, tree, child)

        rolled_tree = jnp.roll(tree, (mutate_idx - end_idx - subtree1_size - subtree2_size - 1), axis=0)
        child = jnp.where((tree_indices < mutate_idx - subtree1_size - subtree2_size) & (tree_indices >= mutate_idx - subtree1_size - subtree2_size - remaining_size), rolled_tree, child)

        child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] + (mutate_idx - end_idx - subtree1_size - subtree2_size - 1), child[:,1:3]))

        child = child.at[mutate_idx, 0].set(operator)
        child = child.at[mutate_idx, 1].set(mutate_idx - 1)
        child = child.at[mutate_idx, 2].set(mutate_idx - subtree1_size - 1)

        subtree1 = jnp.roll(subtree1, -(self.max_nodes - mutate_idx), axis=0)
        subtree1 = subtree1.at[:,1:3].set(jnp.where(subtree1[:,1:3] > -1, subtree1[:,1:3] + (mutate_idx - self.max_nodes), -1))
        child = jnp.where((tree_indices < mutate_idx) & (tree_indices > mutate_idx - subtree1_size - 1), subtree1, child)

        subtree2 = jnp.roll(subtree2, -(self.max_nodes - mutate_idx + subtree1_size), axis=0)
        subtree2 = subtree2.at[:,1:3].set(jnp.where(subtree2[:,1:3] > -1, subtree2[:,1:3] + (mutate_idx - subtree1_size - self.max_nodes), -1))
        child = jnp.where((tree_indices < mutate_idx - subtree1_size) & (tree_indices > mutate_idx - subtree1_size - subtree2_size - 1), subtree2, child)

        return child
    
    def check_invalid_operator_point(self, carry):
        tree, _, mutate_idx, new_operator = carry
        _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree, 1, mutate_idx))

        subtree_size = mutate_idx - end_idx

        empty_nodes = jnp.sum(tree[:,0]==0)
        new_tree_size = jax.lax.select(self.slots[-1*new_operator] == 2, 7, 8)

        return (tree[mutate_idx, 0] == new_operator) | (empty_nodes + subtree_size < new_tree_size)

    def sample_operator_point(self, carry):
        tree, key, _, _ = carry
        key, select_key, sample_key = jr.split(key, 3)

        node_ids = tree[:,0]
        is_operator = (node_ids <= self.func_i) & (node_ids >= self.func_j)
        mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.)
        new_operator = jr.choice(sample_key, a=jnp.arange(self.func_j, self.func_i+1), shape=(), p=jnp.flip(self.expressions[0].operators_prob))

        return (tree, key, mutate_idx, new_operator)

    def mutate_operator(self, tree, key):
        select_key, sample_key, subtree_key = jr.split(key, 3)
        node_ids = tree[:,0]
        is_operator = (node_ids <= self.func_i) & (node_ids >= self.func_j)
        mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.)
        new_operator = jr.choice(sample_key, a=jnp.arange(self.func_j, self.func_i+1), shape=(), p=jnp.flip(self.expressions[0].operators_prob))

        _, _, mutate_idx, new_operator = jax.lax.while_loop(self.check_invalid_operator_point, self.sample_operator_point, (tree, jr.fold_in(key, 0), mutate_idx, new_operator))

        current_slots = self.slots[-1*node_ids[mutate_idx].astype(int)]
        new_slots = self.slots[-1*new_operator]

        child = jax.lax.select(current_slots==2, jax.lax.select(new_slots==2, tree.at[mutate_idx, 0].set(new_operator), self.replace_with_one_subtree(tree, mutate_idx, new_operator, subtree_key)), 
                                   jax.lax.select(new_slots==2, self.replace_with_two_subtrees(tree, mutate_idx, new_operator, subtree_key), tree.at[mutate_idx, 0].set(new_operator)))

        return child

    def delete_operator(self, tree, key):
        tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        select_key, sample_key, float_key, variable_key = jr.split(key, 4)
        node_ids = tree[:,0]
        is_operator = (node_ids <= self.func_i) & (node_ids >= self.func_j)
        is_operator = is_operator.at[-1].set(False)
        delete_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.)
        _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree, 1, delete_idx))

        remaining_size = end_idx - jnp.sum(tree[:,0]==0) + 1

        float = jr.normal(float_key)
        new_leaf = jax.lax.select(jr.uniform(sample_key)<0.5, -1, jr.randint(variable_key, shape=(), minval=self.leaf_j, maxval=self.func_j))

        child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1))
        child = jnp.where(tree_indices > delete_idx, tree, child)

        rolled_tree = jnp.roll(tree, delete_idx - end_idx - 1, axis=0)
        child = jnp.where((tree_indices < delete_idx) & (tree_indices >= delete_idx - remaining_size), rolled_tree, child)
        child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (delete_idx - 1)) & (child[:,1:3] > -1), child[:,1:3] + (delete_idx - end_idx - 1), child[:,1:3]))

        child = child.at[delete_idx, 0].set(new_leaf)
        child = jax.lax.select(new_leaf==-1, child.at[delete_idx, 3].set(float), child.at[delete_idx, 3].set(0))

        return child

    def prepend_operator(self, tree, key):
        tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        sample_key, subtree_key, side_key = jr.split(key, 3)
        new_operator = jr.choice(sample_key, a=jnp.arange(self.func_j, self.func_i+1), shape=(), p=jnp.flip(self.expressions[0].operators_prob))
        new_slots = self.slots[-1*new_operator]
        subtree = self.sample_tree(subtree_key, depth=2)
        subtree_size = jnp.sum(subtree[:,0]!=0)
        tree_size = jnp.sum(tree[:,0]!=0)

        second_branch = jr.bernoulli(side_key)

        child = jnp.roll(tree, -1 - (new_slots - 1) * second_branch*subtree_size, axis=0)
        child = child.at[:,1:3].set(jnp.where(child[:,1:3] > -1, child[:,1:3] - 1 - (new_slots - 1) * second_branch*subtree_size, child[:,1:3]))

        rolled_subtree = jnp.roll(subtree, -1 - (1-second_branch) * tree_size, axis=0)
        rolled_subtree = rolled_subtree.at[:,1:3].set(jnp.where(rolled_subtree[:,1:3] > -1, rolled_subtree[:,1:3] - 1 - (1-second_branch)*tree_size, rolled_subtree[:,1:3]))

        child_2_branches = jax.lax.select(second_branch, jnp.where((tree_indices < self.max_nodes - 1) & (tree_indices >= self.max_nodes - subtree_size - 1), rolled_subtree, child), jnp.where((tree_indices < self.max_nodes - tree_size - 1) & (tree_indices >= self.max_nodes - tree_size - subtree_size - 1), rolled_subtree, child))

        child = jax.lax.select(new_slots==2, child_2_branches, child)
        child = child.at[-1, 0].set(new_operator)
        child = child.at[-1, 1].set(self.max_nodes - 2)
        child = child.at[-1, 2].set(jax.lax.select(new_slots==2, self.max_nodes - jax.lax.select(second_branch, subtree_size, tree_size) - 2, -1))

        return child

    def insert_operator(self, tree, key):
        tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        select_key, sample_key, subtree_key, side_key = jr.split(key, 4)
        node_ids = tree[:,0]
        is_operator = (node_ids <= self.func_i) & (node_ids >= self.func_j)
        is_operator = is_operator.at[-1].set(False)
        mutate_idx = jr.choice(select_key, jnp.arange(tree.shape[0]), p = is_operator*1.)
        _, _, end_idx = jax.lax.while_loop(lambda carry: carry[1]>0, self.find_end_idx, (tree, 1, mutate_idx))

        new_operator = jr.choice(sample_key, a=jnp.arange(self.func_j, self.func_i+1), shape=(), p=jnp.flip(self.expressions[0].operators_prob))
        new_slots = self.slots[-1*new_operator]
        subtree = self.sample_tree(subtree_key, depth=2)
        subtree_size = jnp.sum(subtree[:,0]!=0)
        tree_size = mutate_idx - end_idx

        second_branch = jr.bernoulli(side_key)

        child = jnp.tile(jnp.array([0.0,-1.0,-1.0,0.0]), (self.max_nodes, 1))
        child = jnp.where(tree_indices > mutate_idx, tree, child)
        child = jnp.where(tree_indices < end_idx - (new_slots - 1) * subtree_size, jnp.roll(tree, -(new_slots - 1) * subtree_size - 1, axis=0), child)
        child = child.at[:,1:3].set(jnp.where((child[:,1:3] <= (end_idx)) & (child[:,1:3] > -1), child[:,1:3] - (new_slots - 1) * subtree_size - 1, child[:,1:3]))

        rolled_tree = jnp.roll(tree, - (new_slots - 1) * second_branch * subtree_size - 1, axis=0)
        rolled_tree = rolled_tree.at[:,1:3].set(jnp.where(rolled_tree[:,1:3] > -1, rolled_tree[:,1:3] - 1 - (new_slots - 1) * second_branch*subtree_size, rolled_tree[:,1:3]))

        rolled_subtree = jnp.roll(subtree, mutate_idx - self.max_nodes - (1-second_branch) * tree_size, axis=0)
        rolled_subtree = rolled_subtree.at[:,1:3].set(jnp.where(rolled_subtree[:,1:3] > -1, rolled_subtree[:,1:3] - (self.max_nodes - mutate_idx) - (1-second_branch)*tree_size, rolled_subtree[:,1:3]))

        lower_tree = jax.lax.select(second_branch, jnp.where(tree_indices <= mutate_idx - subtree_size - 1, rolled_tree, rolled_subtree), 
                                jnp.where(tree_indices <= end_idx - 1, rolled_subtree, rolled_tree))
        
        child_2_branches = jnp.where((tree_indices <= mutate_idx - 1) & (tree_indices > mutate_idx - subtree_size - tree_size - 1), lower_tree, child)

        child_1_branch = jnp.where((tree_indices <= mutate_idx - 1) & (tree_indices >= mutate_idx - tree_size), rolled_tree, child)
        
        child = jax.lax.select(new_slots==2, child_2_branches, child_1_branch)
        child = child.at[mutate_idx, 0].set(new_operator)
        child = child.at[mutate_idx, 1].set(mutate_idx - 1)
        child = child.at[mutate_idx, 2].set(jax.lax.select(new_slots==2, mutate_idx - jax.lax.select(second_branch, subtree_size, tree_size) - 1, -1))

        return child

    def replace_tree(self, tree, key):
        return self.sample_tree(key, self.max_init_depth)

    def sample_pair(self, parent1, parent2, keys, reproduction_probability):
        offspring = jax.vmap(self.sample_trees, in_axes=[1, None])(keys, self.max_init_depth)
        return offspring[0], offspring[1]
    
    def mutate_tree(self, tree, key, mutate_function):
        return jax.lax.switch(mutate_function, self.mutate_functions, tree, key)
    
    def get_mutations(self, tree, key):
        [self.add_subtree, self.mutate_leaf, self.mutate_operator, self.delete_operator, self.prepend_operator, self.insert_operator, self.replace_tree]
        mutation_probs = jnp.ones(len(self.mutate_functions))
        [self.add_subtree, self.mutate_leaf, self.mutate_operator, self.delete_operator, self.prepend_operator, self.insert_operator, self.replace_tree]
        mutation_probs = jax.lax.select(jnp.sum(tree[:,0]==0) < 8, jnp.array([0., 1., 1., 1., 0., 0., 1.]), mutation_probs)
        mutation_probs = jax.lax.select(jnp.sum(tree[:,0]!=0) <= 3, jnp.array([1., 1., 1., 0., 1., 0., 1.]), mutation_probs)
        mutation_probs = jax.lax.select(jnp.sum(tree[:,0]!=0) == 1, jnp.array([1., 1., 0., 0., 1., 0., 1.]), mutation_probs)
        
        return jr.choice(key, jnp.arange(len(self.mutate_functions)), p=mutation_probs)
    
    def mutate_trees(self, trees, keys, reproduction_probability):
        index_key, func_key = jr.split(keys[0])
        _, mutate_indices, _ = jax.lax.while_loop(lambda carry: jnp.sum(carry[1])==0, self.sample_indices, (index_key, jnp.zeros(trees.shape[0]), reproduction_probability))
        mutate_functions = jax.vmap(self.get_mutations)(trees, keys)

        mutated_trees = jax.vmap(self.mutate_tree)(trees, keys, mutate_functions)

        return jnp.where(mutate_indices[:,None,None] * jnp.ones_like(trees), mutated_trees, trees)
    
    def mutate_pair(self, parent1, parent2, keys, reproduction_probability):
        child1 = self.mutate_trees(parent1, keys[:,0], reproduction_probability)
        child2 = self.mutate_trees(parent2, keys[:,1], reproduction_probability)
        return child1, child2

    def evolve_trees(self, parent1, parent2, keys, type, reproduction_probability):
        child1, child2 = jax.lax.switch(type, [self.crossover_trees, self.mutate_pair, self.sample_pair], parent1, parent2, keys, reproduction_probability)

        return child1, child2

    def tournament_selection(self, population, fitness, key, tournament_probabilities):
        tournament_key, winner_key = jr.split(key)
        indices = jr.choice(tournament_key, jnp.arange(self.population_size), shape=(self.tournament_size,))
        index = jr.choice(winner_key, indices[jnp.argsort(fitness[indices])], p=tournament_probabilities)
        return population[index]
    
    def check_restart_population(self, key, best_fitness, restart_iter_threshold, last_restart):
        new_population = self.sample_population(key)
        restart = (last_restart > restart_iter_threshold) & (best_fitness[self.current_generation] >= best_fitness[self.current_generation - restart_iter_threshold])
        last_restart = jax.lax.select(restart, 0., last_restart + 1)
        return restart, last_restart, new_population
    
    def evolve_population(self, population, fitness, key, reproduction_type_probabilities, reproduction_probability, tournament_probabilities):
        left_key, right_key, repro_key, cx_key = jr.split(key, 4)
        # restart, last_restart, restart_population = self.check_restart_population(restart_key, best_fitness, restart_iter_threshold, last_restart)
        left_parents = jax.vmap(self.tournament_selection, in_axes=[None, None, 0, None])(population, fitness, jr.split(left_key, self.population_size//2), tournament_probabilities)
        right_parents = jax.vmap(self.tournament_selection, in_axes=[None, None, 0, None])(population, fitness, jr.split(right_key, self.population_size//2), tournament_probabilities)
        reproduction_type = jr.choice(repro_key, jnp.arange(3), shape=(self.population_size//2,), p=reproduction_type_probabilities)
        left_children, right_children = jax.vmap(self.evolve_trees, in_axes=[0, 0, 0, 0, None])(left_parents, right_parents, jr.split(cx_key, (self.population_size//2, self.num_trees, 2)), reproduction_type, reproduction_probability)
        # evolved_population = jax.lax.select(restart, restart_population, jnp.concatenate([left_children, right_children], axis=0))
        evolved_population = jnp.concatenate([left_children, right_children], axis=0)
        return evolved_population
    
    def migrate_population(self, receiver, sender, receiver_fitness, sender_fitness, migration_size):
        population_indices = jnp.arange(self.population_size)
        sorted_receiver = receiver[jnp.argsort(receiver_fitness, descending=True)]
        sorted_sender = sender[jnp.argsort(sender_fitness, descending=False)]
        return jnp.where((population_indices < migration_size)[:,None,None,None], sorted_sender, sorted_receiver)
    
    def evolve(self, populations, fitness, key):
        populations = jax.lax.select((self.num_populations > 1) & (((self.current_generation+1)%self.migration_period) == 0), 
                                     jax.vmap(self.migrate_population, in_axes=[0, 0, 0, 0, None])(populations, jnp.roll(populations, 1, axis=0), fitness, jnp.roll(fitness, 1, axis=0), self.migration_size), 
                                     populations)
        new_population = jax.vmap(self.jit_evolve_population, in_axes=[0, 0, 0, 0, 0, 0])(populations, fitness, jr.split(key, self.num_populations), self.reproduction_type_probabilities, 
                            self.reproduction_probabilities, self.tournament_probabilities)
        self.current_generation += 1
        return new_population