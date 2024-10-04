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

def eval_rastrigin(individual):
    err = 0.0
    for i in range(len(individual)):
        xi = individual[i]
        err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return (err,)

def GA():
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -5.12, 5.12)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_rastrigin)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def parse_tuple(arg):
    return tuple(map(float, arg.strip("()").split(',')))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=str, required=True, help="Population size in format '(min,max)'")
    ap.add_argument("--pc", type=str, required=True, help="Probability of crossover in format '(min,max)'")
    ap.add_argument("--pm", type=str, required=True, help="Probability of mutation in format '(min,max)'")
    ap.add_argument("--g", type=str, required=True, help="Number of generations in format '(min,max)'")

    args = vars(ap.parse_args())

    toolb = GA()

    pops_range = parse_tuple(args["pop"])
    probc_range = parse_tuple(args["pc"])
    probm_range = parse_tuple(args["pm"])
    g_range = parse_tuple(args["g"])

    pops_step = 50
    prob_step = 0.1
    g_step = 20

    total_iters = (
        len(np.arange(pops_range[0], pops_range[1] + pops_step, pops_step)) *
        len(np.arange(probc_range[0], probc_range[1] + prob_step, prob_step)) *
        len(np.arange(probm_range[0], probm_range[1] + prob_step, prob_step)) *
        len(np.arange(g_range[0], g_range[1] + g_step, g_step))
    )

    best_configs = []
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

                        # Evolución con registro del fitness por generación
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
                            fitnesses = map(toolb.evaluate, invalid_ind)
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

                        best_configs.append({
                            "Population Size": pop_size,
                            "Crossover Probability": prob_crossover,
                            "Mutation Probability": prob_mutation,
                            "Generations": generation_size,
                            "Best Fitness": hof[0].fitness.values[0],
                            "Computation Time (s)": elapsed_time
                        })
                        best_configs = sorted(best_configs, key=lambda x: x["Best Fitness"])[:6]

                        pbar.update(1)

    df = pd.DataFrame(best_configs)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"summary_{date_str}.csv", index=False)

    plt.plot(best_fitness_per_gen, label='Best Fitness (Error)')
    plt.title('Evolution of the Best Individual')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Error)')
    plt.legend()
    plt.grid()
    plt.savefig(f"fitness_evolution_{date_str}.png")
    plt.show(block=False)


    best_config = best_configs[0]  
    print("\nBest individual configuration:")
    print(f"Population Size: {best_config['Population Size']}")
    print(f"Crossover Probability: {best_config['Crossover Probability']}")
    print(f"Mutation Probability: {best_config['Mutation Probability']}")
    print(f"Generations: {best_config['Generations']}")
    print(f"Best Fitness: {best_config['Best Fitness']}")
    print(f"Computation Time (s): {best_config['Computation Time (s)']}")
