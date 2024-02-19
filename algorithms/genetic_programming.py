import warnings
import jax
import time
import jax.numpy as jnp
import jax.random as jrandom
from pathos.multiprocessing import ProcessingPool as Pool

import genetic_operators.reproduction as reproduction
import miscellaneous.helper_functions as helper_functions
import evaluators.evaluate as evaluate
import genetic_operators.migration as migration
import genetic_operators.simplification as simplification

def run(env, expressions, layer_sizes, population_size, num_populations, num_generations, T, dt, pool_size, param_setting: str, n_seeds: int = 1, start_seed: int = 0):
    with warnings.catch_warnings(action="ignore"):
        restart_iter_threshold: int = jnp.linspace(10, 4, num_populations, dtype=int)

    migration_period: int = 5
    migration_method: str = "ring"
    migration_percentage: float = 0.1
    migration_size = int(migration_percentage*population_size)

    best_fitnesses = jnp.zeros((n_seeds, num_generations))
    # start=time.time()
    best_solutions = []
    # rs_best_fitnesses = []
    # rs_best_solutions = []
    LQG_fitnesses = jnp.zeros((n_seeds))

    for seed in range(n_seeds):
        key = jrandom.PRNGKey(seed + start_seed)
        key, data_key, test_data_key = jrandom.split(key, 3)

        evaluator = evaluate.Evaluator(env, expressions, layer_sizes, data_key, dt, T, param_setting=param_setting)
        y_key, a_key, u_key, target_key = jrandom.split(test_data_key, 4)
        test_samples = {"y":jrandom.permutation(y_key, jnp.repeat(jnp.arange(-2,3)[:,None], env.n_obs, axis=1), axis=0, independent=True), 
                        "a":jrandom.permutation(a_key, jnp.repeat(jnp.arange(-2,3)[:,None], layer_sizes[0], axis=1), axis=0, independent=True), #iets aan doen
                        "u":jrandom.permutation(u_key, jnp.repeat(jnp.arange(-2,3)[:,None], env.n_control, axis=1), axis=0, independent=True), 
                        "target":jrandom.permutation(target_key, jnp.repeat(jnp.arange(-2,3)[:,None], env.n_targets, axis=1), axis=0, independent=True)}
        reproducer = reproduction.Reproducer(expressions, layer_sizes, population_size, num_populations, test_samples)

        # LQG_fitness = evaluator.LQG_fitness()
        # print(f"Seed {seed+start_seed}, LQG fitness = {LQG_fitness}")
        # LQG_fitnesses = LQG_fitnesses.at[seed].set(LQG_fitness)

        pool = Pool(pool_size)
        pool.close()

        best_solution = None
        best_fitness = jnp.inf
        best_fitness_per_population = jnp.zeros((num_generations, num_populations))

        last_restart = jnp.zeros(num_populations)

        #Initialize new population
        new_keys = jrandom.split(key, num_populations+1)
        key = new_keys[0]

        pool.restart()
        populations = pool.amap(lambda x: reproducer.sample_trees(new_keys[x]), range(num_populations))
        pool.close()
        pool.join()
        populations = populations.get()

        for g in range(num_generations):
            # end = time.time()
            # print("sampled ",end-start)
            # start = end
            pool.restart()
            fitness = pool.amap(lambda x: evaluator.get_fitness(x), helper_functions.flatten(populations)) #Evaluate each solution parallely on a pool of workers
            pool.close()
            pool.join()
            tries = 0
            while not fitness.ready():
                time.sleep(1)
                tries += 1
                print(tries)
                if tries >= 2:
                    print("TIMEOUT")
                    break

            flat_fitnesses = jnp.array(fitness.get())
            fitnesses = jnp.reshape(flat_fitnesses,(num_populations,population_size))
            # print(fitnesses)
            # print(jnp.mean(fitnesses, axis=1),jnp.min(fitnesses, axis=1))

            best_fitness_per_population = best_fitness_per_population.at[g].set(jnp.min(fitnesses, axis=1))
            
            #Set the fitness of each solution
            for pop in range(num_populations):
                population = populations[pop]
                for candidate in range(population_size):
                    population[candidate].set_fitness(fitnesses[pop,candidate])

            best_solution_of_g = helper_functions.best_solution(populations, fitnesses)
            best_fitness_at_g = evaluator.get_fitness(best_solution_of_g, add_regularization=False)
            if best_fitness_at_g < best_fitness:
                best_fitness = best_fitness_at_g
                best_solution = best_solution_of_g
            best_fitnesses = best_fitnesses.at[seed, g].set(best_fitness)

            # end = time.time()
            # print("eval ", end-start)
            # start = end

            if best_fitnesses[seed, g]==0.0: #A solution reached a satisfactory score
                print(f"Converge settings satisfied, best_fitness: {best_fitness}, best solution: {simplification.trees_to_sympy(best_solution()[0])}, readout: {simplification.trees_to_sympy(best_solution()[1])}")
                best_fitnesses = best_fitnesses.at[seed, g:].set(best_fitness)
                best_solutions.append(best_solution)

                break

            if g < num_generations-1: #The final generation has not been reached yet, so a new population is sampled
                print(f"In generation {g+1}, best_fitness: {best_fitness}, best solution: {simplification.trees_to_sympy(best_solution()[0])}, readout: {simplification.trees_to_sympy(best_solution()[1])}")

                #Migrate individuals between populations every few generations
                if ((g+1)%migration_period)==0:
                    key, new_key = jrandom.split(key)
                    populations = migration.migrate_populations(populations, migration_method, migration_size, new_key)

                last_restart = last_restart + 1
                restart = jnp.logical_and(last_restart>restart_iter_threshold, best_fitness_per_population[g] >= jnp.array([best_fitness_per_population[g-restart_iter_threshold[i], i] for i in range(num_populations)]))
                last_restart = last_restart.at[restart].set(0)

                def update_population(index, population, restart, key):
                    if restart:
                        return reproducer.sample_trees(key)
                    else:
                        return reproducer.next_population(population, index, key)
                new_keys = jrandom.split(key, num_populations+1)
                key = new_keys[0]
                pool.restart()
                new_populations = pool.amap(lambda x: update_population(x, populations[x], restart[x], new_keys[1+x]), range(num_populations)) #Evaluate each solution parallely on a pool of workers
                pool.close()

                populations = new_populations.get()

            else: #Final generation is reached
                best_solutions.append(best_solution)
                print(f"Final generation, best_fitness: {best_fitness}, best solution: {simplification.trees_to_sympy(best_solution()[0])}, readout: {simplification.trees_to_sympy(best_solution()[1])}")
        
    return best_fitnesses, best_solutions, LQG_fitnesses