import operator
import math
import random
import numpy
from functools import partial
from deap import algorithms, base, creator, tools, gp
import argparse
from utils import plotting
import importlib
import time
import datetime
import os

def load_error_function(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def gen_random_points(func, n_points=20, x_range=(-100, 100)):
    points = []
    for _ in range(n_points):
        x = random.uniform(*x_range)
        y = func(x)
        points.append((x, y))
    return points

#protected functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedSqrt(x):
    return math.sqrt(x) if x >= 0 else 0.0

def protectedLog(x):
    return math.log(x) if x > 0 else 0.0
    
def protectedExp(x):
    try:
        return math.exp(x) if x < 700 else 1e10  # Limita el valor máximo
    except OverflowError:
        return 1e10  # Valor seguro en caso de error

def protectedTan(x):
    try:
        result = math.tan(x)
        return result if abs(result) < 1e10 else 0.0
    except ValueError:
        return 0.0

#eval
def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    sqerrors = []
    for x, y in points:
        try:
            # Limita el valor de la función evaluada para evitar sobrecargas
            fx = func(x)
            if abs(fx) > 1e10:  # Evita valores excesivamente grandes
                fx = 1e10 if fx > 0 else -1e10
            sqerrors.append((fx - y) ** 2)
        except OverflowError:
            # Si ocurre un error, asigna un valor de error muy grande
            sqerrors.append(1e10)
    return math.fsum(sqerrors) / len(points),

#global
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

# new functions
pset.addPrimitive(protectedTan, 1)      # Tangente protegida
pset.addPrimitive(protectedLog, 1)      # Logaritmo natural protegido
pset.addPrimitive(protectedSqrt, 1)     # Raíz cuadrada protegida
pset.addPrimitive(protectedExp, 1)      # Exponencial protegida
pset.addPrimitive(abs, 1)               # Valor absoluto

pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def init_toolbox(func):
    toolbox.register("evaluate", evalSymbReg, points=gen_random_points(func))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    # limit tree height
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxb", type=float, required=True, help="cxb")
    ap.add_argument("--mut", type=float, required=True, help="mutation")
    ap.add_argument("--gen", type=int, required=True, help="Generations")
    ap.add_argument("--output", type=str, required=True, help="Output Path")
    ap.add_argument("--function", type=str, required=True, help="Error function name")
    
    args = vars(ap.parse_args())
    cxb = args["cxb"]
    mut = args["mut"]
    gen = args["gen"]
    function_name = args["function"]
    output_path = args["output"]

    os.makedirs(output_path, exist_ok=True)

    error_function = load_error_function("error_functions", function_name)
    
    init_toolbox(error_function)

    random.seed(318)
    points = gen_random_points(error_function)
    toolbox.register("evaluate", evalSymbReg, points=points)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxb, mut, gen, stats=mstats,
                                   halloffame=hof, verbose=True)
    # plot better
    best_individual = hof[0]
    best_func = toolbox.compile(expr=best_individual)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plotting.plot_function(points, best_func, error_function, output_path, date_str)
    plotting.plot_tree(best_individual, output_path, date_str)
