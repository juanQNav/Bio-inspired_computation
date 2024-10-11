import os
import random
import numpy as np
import math
from deap import base, creator, tools, algorithms
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import evals as ev
from utils import gest_data, plotting
import importlib
import inspect

def load_eval_function(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def GA(space_sol, eval, eval_args=None, n=10):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, space_sol[0], space_sol[1])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Detect the number of arguments required by the evaluation function
    eval_signature = inspect.signature(eval)
    num_params = len(eval_signature.parameters)

    if num_params == 1:
        # Only the individual is required
        toolbox.register("evaluate", eval)
    else:
        # Other arguments are required (e.g., weights and values)
        toolbox.register("evaluate", lambda ind: eval(ind, *eval_args))

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def binarize_ind(individual):
   return [1 if x > 0.5 else 0 for x in individual]

def parse_tuple(arg):
    return tuple(map(float, arg.strip("()").split(',')))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=str, required=True, help="Population size in format '(min,max)'")
    ap.add_argument("--pc", type=str, required=True, help="Probability of crossover in format '(min,max)'")
    ap.add_argument("--pm", type=str, required=True, help="Probability of mutation in format '(min,max)'")
    ap.add_argument("--g", type=str, required=True, help="Number of generations in format '(min,max)'")
    ap.add_argument("--ev", type=str, required=False, help="Error function name")
    ap.add_argument("--sp", type=str, required=True, help="Space solution '(min,max)'")
    ap.add_argument("--dp", type=str, required=False, help="Data path csv")
    ap.add_argument("--output", type=str, required=True, help="Output path")
    ap.add_argument("--bin", action="store_true", help="Show vector binary")

    args = vars(ap.parse_args())

    pops_range = parse_tuple(args["pop"])
    probc_range = parse_tuple(args["pc"])
    probm_range = parse_tuple(args["pm"])
    g_range = parse_tuple(args["g"])
    space_sol = parse_tuple(args["sp"])
    eval_name = args["ev"]
    output_path = args["output"]
    binary = args['bin']
    data_path = args['dp']

    os.makedirs(output_path, exist_ok=True)


    eval_function = load_eval_function("evals", eval_name)

    eval_args = []
    if eval_name == "eval_knapsack":
        weights, values = gest_data.load_knapsack_data(data_path)
        eval_args = [weights, values]
        n = len(weights)


    toolb = GA(space_sol=space_sol, eval=eval_function, eval_args=eval_args, n = n)

    pops_step = 50
    prob_step = 0.1
    g_step = 1

    total_iters = (
        len(np.arange(pops_range[0], pops_range[1] + pops_step, pops_step)) *
        len(np.arange(probc_range[0], probc_range[1] + prob_step, prob_step)) *
        len(np.arange(probm_range[0], probm_range[1] + prob_step, prob_step)) *
        len(np.arange(g_range[0], g_range[1] + g_step, g_step))
    )

    sumary_configs = []
    best_fitness_per_gen = []

    with tqdm(total=total_iters) as pbar:
        for pop_size in np.arange(pops_range[0], pops_range[1] + pops_step, pops_step):
            for prob_crossover in np.arange(probc_range[0], probc_range[1] + prob_step, prob_step):
                for prob_mutation in np.arange(probm_range[0], probm_range[1] + prob_step, prob_step):
                    for generation_size in np.arange(g_range[0], g_range[1] + g_step, g_step):
                        start_time = time.time()
                        population = toolb.population(n=int(pop_size))

                        stats = tools.Statistics(lambda ind: ind.fitness.values)
                        stats.register("avg", np.mean)
                        stats.register("std", np.std)
                        stats.register("min", np.min)
                        stats.register("max", np.max)

                        hof = tools.HallOfFame(1)

                        fitness_per_gen = []

                        for gen in range(int(generation_size)):
                            offspring = toolb.select(population, len(population))
                            offspring = list(map(toolb.clone, offspring))

                            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                                if random.random() < prob_crossover:
                                    toolb.mate(child1, child2)
                                    del child1.fitness.values
                                    del child2.fitness.values

                            for mutant in offspring:
                                if random.random() < prob_mutation:
                                    toolb.mutate(mutant)
                                    del mutant.fitness.values

                            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                            fitnesses = map(lambda ind: toolb.evaluate(ind), invalid_ind)
                            for ind, fit in zip(invalid_ind, fitnesses):
                                ind.fitness.values = fit

                            population[:] = offspring

                            fits = [ind.fitness.values[0] for ind in population]
                            best_fit_gen = min(fits)
                            fitness_per_gen.append(best_fit_gen)

                            hof.update(population)

                        best_fitness_per_gen = fitness_per_gen 
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        
                        solution = hof[0]
                        if binary:
                            solution = '  '.join(f"{val:.4f}" for val in binarize_ind(hof[0]))

                        sumary_configs.append({
                            "Population Size": pop_size,
                            "Crossover Probability": prob_crossover,
                            "Mutation Probability": prob_mutation,
                            "Generations": generation_size,
                            "Best Fitness": hof[0].fitness.values[0],
                            "Best solution": solution,
                            "Computation Time (s)": elapsed_time
                        })
                        sumary_configs = sorted(sumary_configs, key=lambda x: x["Best Fitness"])

                        top_two = sumary_configs[:2]

                        remaining_configs = sumary_configs[len(sumary_configs)//2:]
                        num_random = min(4, len(remaining_configs))

                        random_others = random.sample(remaining_configs, num_random)

                        combined_results = top_two + random_others

                        combined_results = sorted(combined_results, key=lambda x: x["Best Fitness"])
                        sumary_configs = combined_results
                        pbar.update(1)

    df = pd.DataFrame(sumary_configs)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(os.path.join(output_path,f"summary_{date_str}.csv"), index=False)

    plotting.plot_errors(best_fitness_per_gen, date_str, output_path)

    best_config = sumary_configs[0]  
    print("\nBest individual configuration:")
    print(f"Population Size: {best_config['Population Size']}")
    print(f"Crossover Probability: {best_config['Crossover Probability']}")
    print(f"Mutation Probability: {best_config['Mutation Probability']}")
    print(f"Generations: {best_config['Generations']}")
    print(f"Best Fitness: {best_config['Best Fitness']}")
    print(f"Best solution found (binary): {best_config['Best solution']}")
    print(f"Computation Time (s): {best_config['Computation Time (s)']}")
