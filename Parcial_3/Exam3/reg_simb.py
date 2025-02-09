import operator
import math
import random
import numpy as np
from functools import partial
from deap import algorithms, base, creator, tools, gp
import argparse
from utils import plotting
from utils import protected_funcs as pf
from utils.points import split_data, gen_random_points
import importlib
import time
import datetime
import os
import itertools
import pandas as pd
from tqdm import tqdm

# Function to load the error function
def load_error_function(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

# Function to evaluate the symbolic regression
def evalSymbReg(individual, points, pset):
    func = gp.compile(expr=individual, pset=pset)
    sqerrors = []
    for x, y in points:
        try:
            fx = func(x)
            if abs(fx) > 1e10:
                fx = 1e10 if fx > 0 else -1e10
            sqerrors.append((fx - y) ** 2)
        except OverflowError:
            sqerrors.append(1e10)
    return np.sum(sqerrors) / len(points),

# Create creator
def create_creator():
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Primitive set
def create_pset():
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(pf.protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(pf.protectedExp, 1)
    # pset.addPrimitive(pf.protectedTan, 1)
    # pset.addPrimitive(pf.protectedLog, 1)
    # pset.addPrimitive(pf.protectedSqrt, 1)
    # pset.addPrimitive(abs, 1)

    pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
    pset.renameArguments(ARG0='x')
    return pset

# Create toolbox
def create_toolbox(pset, trian_points):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalSymbReg, pset=pset, points=train_points)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    return toolbox

# run evolution algorithm
def run_evolution(toolbox, cxb, mut, gen,n_pop, hof):
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=n_pop)
    pop, log = algorithms.eaSimple(pop, toolbox, cxb, mut, gen, stats=mstats,
                                   halloffame=hof, verbose=False)
    return pop, log

def parse_tuple(arg, is_float=True):

    return tuple(map(float, arg.strip("()").split(','))) if is_float else tuple(map(int, arg.strip("()").split(',')))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxb", type=str, required=True, help="Crossover probability format (min, max)")
    ap.add_argument("--mut", type=str, required=True, help="Mutation probability format (min, max)")
    ap.add_argument("--gen", type=str, required=True, help="Number of generations format (min, max)")
    ap.add_argument("--npop", type=str, required=True, help="Population size format (min, max)")
    ap.add_argument("--function", type=str, required=True, help="Error function name")
    ap.add_argument("--output", type=str, required=True, help="Output path")
    ap.add_argument("--label_func", type=str, required=True, help="Label of the error function")
    args = vars(ap.parse_args())

    CXB_range = parse_tuple(args["cxb"])
    MUT_range = parse_tuple(args["mut"])
    GEN_range = parse_tuple(args["gen"], is_float=False)
    NPOP_range = parse_tuple(args["npop"], is_float=False)
    FUNCTION_NAME = args["function"]
    OUTPUT_PATH = args["output"]
    LABEL_FUNC = args["label_func"]

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print(CXB_range, MUT_range, GEN_range, NPOP_range)

    param_combinations = list(itertools.product(np.arange(CXB_range[0], CXB_range[1] + 0.1, 0.2),
                                                np.arange(MUT_range[0], MUT_range[1] + 0.1, 0.2),
                                                np.arange(GEN_range[0], GEN_range[1] + 1, 2),
                                                np.arange(NPOP_range[0], NPOP_range[1] + 1, 25)))

    error_function = load_error_function("error_functions", FUNCTION_NAME)

    summary = []

    for i, (cxb, mut, gen, npop) in enumerate(tqdm(param_combinations, desc="Running experiments")):
        start_time = time.time()

        create_creator()
        pset = create_pset()

        points = gen_random_points(error_function)

        train_points, test_points = split_data(points, train_ratio=0.7)    
        toolbox = create_toolbox(pset, train_points)

        hof = tools.HallOfFame(1)
        pop, log = run_evolution(toolbox, cxb, mut, gen, npop, hof)

        end_time = time.time()
        exec_time = end_time - start_time

        best_individual = hof[0]
        best_func = gp.compile(expr=best_individual, pset=pset)
        test_error = evalSymbReg(hof[0], test_points, pset)

        test_error_value = float(test_error[0])

        func_str = str(best_individual)

        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        plotting.plot_function(train_points, test_points, best_func, f"{LABEL_FUNC} iteration_{i}", OUTPUT_PATH, date_str, view=False)
        plotting.plot_tree(best_individual, OUTPUT_PATH, f"{date_str}_{i}", view=False)

        summary.append({
            "Experiment": i,
            "Crossover": cxb,
            "Mutation": mut,
            "Generations": gen,
            "Population": npop,
            "Best Function": func_str,
            "Error": round(test_error_value, 5),  
            "Execution Time": round(exec_time, 5) 
        })

    results = pd.DataFrame(summary)
    results = results.sort_values(by="Error", ascending=True)
    results = results.reset_index(drop=True)
    print(results.head(10))
    results.to_csv(os.path.join(OUTPUT_PATH, "summary_results.csv"), index=False)

    print("All experiments completed.")