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
                 migration_method: str = "ring", 
                 migration_period: int = 5,
                 leaf_sd: float = 1.0, 
                 migration_percentage: float = 0.1,
                 restart_iter_threshold: int = None,
                 gradient_optimisation: bool = False,
                 gradient_steps: int = 10) -> None:
        super().__init__(num_generations, population_size, fitness_function)
        # assert len(layer_sizes) == len(expressions), "There is not a set of expressions for every type of layer"
        self.expressions = expressions
        self.layer_sizes = layer_sizes
        self.num_populations = num_populations
        self.max_depth = max_depth
        self.max_init_depth = max_init_depth
        self.max_nodes = min(max_nodes, 2**self.max_depth-1)
        self.init_method = init_method
        if (init_method=="ramped") or (init_method=="full"):
            assert 2**self.max_init_depth < self.max_nodes, "Full trees are not possible given initialization depth and max size"

        self.size_parsinomy = size_parsinomy
        self.leaf_sd = leaf_sd

        self.migration_period = migration_period
        self.migration_method = migration_method
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
        
        self.reproduction_type_probabilities = jnp.vstack([jnp.linspace(0.9,0.5,self.num_populations),jnp.linspace(0.1,0.5,self.num_populations),
                                         jnp.linspace(0.0,0.0,self.num_populations),jnp.linspace(0.0,0.0,self.num_populations)]).T
        self.reproduction_probabilities = jnp.linspace(1.0, 0.5, self.num_populations)
        self.elite_percentage = jnp.linspace(0.04, 0.04)

        self.mutation_probabilities = {}
        self.mutation_probabilities["mutate_operator"] = 0.5
        self.mutation_probabilities["delete_operator"] = 0.5
        self.mutation_probabilities["insert_operator"] = 1.0
        self.mutation_probabilities["mutate_constant"] = 0.1
        self.mutation_probabilities["mutate_leaf"] = 0.5
        self.mutation_probabilities["sample_subtree"] = 1.0
        self.mutation_probabilities["prepend_operator"] = 1.0
        self.mutation_probabilities["add_subtree"] = 1.0

        self.best_fitness_per_population = jnp.zeros((num_generations, self.num_populations))

        self.optimizer = optax.adam(learning_rate=0.01, b1=0.8, b2=0.9)

        func_dict = {}
        func_to_string = {}
        functions = [lambda x, y, _data: 0.0, lambda x, y, _data: 0.0]

        self.map_b_to_d = self.create_map_b_to_d(self.max_depth)

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
        key, matrix, open_slots = carry
        float_key, leaf_key, variable_key, node_key, func_key = jr.split(key, 5)
        _i = self.map_b_to_d[i].astype(int)

        depth = 0.7**(jnp.log(i+1)/jnp.log(2)).astype(int)
        float = jr.normal(float_key)
        leaf = jax.lax.select(jr.uniform(leaf_key)<0.5, -1, jr.randint(variable_key, shape=(), minval=self.leaf_j, maxval=self.func_j))
        index = jax.lax.select((open_slots < self.max_nodes - i - 1) & (depth<=self.max_init_depth), jax.lax.select(jr.uniform(node_key)<(depth), jr.choice(func_key, a=jnp.arange(self.func_j, self.func_i+1), shape=(), p=jnp.flip(self.expressions[0].operators_prob)), leaf), leaf)
        index = jax.lax.select(open_slots == 0, 0, index)
        index = jax.lax.select(i>0, jax.lax.select((self.slots[-jnp.minimum(matrix[self.map_b_to_d[(i + (i%2) - 2)//2].astype(int), 0], 0).astype(int)] + i%2) > 1, index, 0), index)

        matrix = jax.lax.select(self.slots[-1*index] > 0, matrix.at[_i, 1].set(self.map_b_to_d[2*i+1]), matrix.at[_i, 1].set(-1))
        matrix = jax.lax.select(self.slots[-1*index] > 1, matrix.at[_i, 2].set(self.map_b_to_d[2*i+2]), matrix.at[_i, 2].set(-1))

        matrix = jax.lax.select(index == -1, matrix.at[_i,3].set(float), matrix)
        matrix = matrix.at[_i, 0].set(index)

        open_slots = jax.lax.select(index == 0, open_slots, jnp.maximum(0, open_slots + self.slots[-1*index] - 1))

        return (jr.fold_in(key, i), matrix, open_slots)
    
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

    def sample_tree(self, key):
        tree = jax.lax.fori_loop(0, 2**self.max_depth-1, self.sample_FS_node, (key, jnp.zeros((2**self.max_depth-1, 4)), 1))[1]
        return self.prune_tree(tree)
    
    def sample_trees(self, key):
        return jax.vmap(self.sample_tree)(jr.split(key, jnp.sum(self.layer_sizes).item()))
    
    def sample_population(self, key):
        return jax.vmap(self.sample_trees)(jr.split(key, self.population_size))
        
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
        
        start = time.time()
        flat_populations = populations.reshape(self.num_populations*self.population_size, *populations.shape[2:])
        flat_populations = jax.device_put(flat_populations, NamedSharding(mesh, P('i')))
        
        fitness, optimised_population = shard_optimise(flat_populations, data) if self.gradient_optimisation else shard_eval(flat_populations, data)
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
            
        self.best_solutions.append(best_solution)
        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)
        # print("eval", time.time() - start)

        return fitness, optimised_population.reshape((self.num_populations, self.population_size, *optimised_population.shape[1:]))
    
    def increment(self):
        self.current_generation += 1
            
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
    
    def find_end_idx(self, carry):
        tree, openslots, counter = carry
        _, idx1, idx2, _ = tree[counter]
        openslots -= 1
        openslots = jax.lax.select(idx1 < 0, openslots, openslots+1)
        openslots = jax.lax.select(idx2 < 0, openslots, openslots+1)
        counter -= 1
        return (tree, openslots, counter)
    
    def crossover(self, tree1, tree2, key):
        #Define indices of the nodes
        tree_indices = jnp.tile(jnp.arange(self.max_nodes)[:,None], reps=(1,4))
        key1, key2 = jr.split(key)

        #Define last node in tree
        last_node_idx1 = jnp.sum(tree1[:,0]==0)
        last_node_idx2 = jnp.sum(tree2[:,0]==0)

        #Randomly select nodes for crossover
        node_idx1 = jr.randint(key1, shape=(), minval=last_node_idx1, maxval=self.max_nodes)
        node_idx2 = jr.randint(key2, shape=(), minval=last_node_idx2, maxval=self.max_nodes)

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
        child1 = jnp.where((tree_indices >= jax.lax.max(0, node_idx1 + 1 - subtree_size2 - end_idx1 - 1)) & (tree_indices < node_idx1 + 1 - subtree_size2), rolled_tree1, child1)
        child2 = jnp.where((tree_indices >= jax.lax.max(0, node_idx2 + 1 - subtree_size1 - end_idx2 - 1)) & (tree_indices < node_idx2 + 1 - subtree_size1), rolled_tree2, child2)

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
    
    def crossover_trees(self, parent1, parent2, key):
        return jax.vmap(self.crossover)(parent1, parent2, key)

    def tournament_selection(self, population, fitness, key):
        tournament_key, winner_key = jr.split(key)
        indices = jr.choice(tournament_key, jnp.arange(self.population_size), shape=(self.tournament_size,))
        index = jr.choice(winner_key, indices, p=1/(fitness[indices]))
        return population[index]
    
    def evolve_population(self, population, fitness, key):
        left_key, right_key, cx_key = jr.split(key, 3)
        left_parents = jax.vmap(self.tournament_selection, in_axes=[None, None, 0])(population, fitness, jr.split(left_key, self.population_size//2))
        right_parents = jax.vmap(self.tournament_selection, in_axes=[None, None, 0])(population, fitness, jr.split(right_key, self.population_size//2))
        left_children, right_children = jax.vmap(self.crossover_trees)(left_parents, right_parents, jr.split(cx_key, (self.population_size//2, jnp.sum(self.layer_sizes).item())))
        return jnp.concatenate([left_children, right_children], axis=0)

    def evolve(self, populations, fitness, key):
        new_population = jax.vmap(self.evolve_population)(populations, fitness, jr.split(key, self.num_populations))
        return new_population