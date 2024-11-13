import numpy
import matplotlib.pyplot as plt
import os
from deap import gp
import graphviz

#plot function
def plot_function(points, best_func, func_obj, output, date_str):
    x_vals = [x for x, y in points]
    y_vals = [y for x, y in points]
    x_range = numpy.linspace(min(x_vals), max(x_vals), 100)
    y_real = [func_obj(x) for x in x_range]
    y_best = [best_func(x) for x in x_range]

    plt.scatter(x_vals, y_vals, color='red', label="Puntos aleatorios (x, y)")
    plt.plot(x_range, y_real, label="y = x^2 / 2", color='blue')
    plt.plot(x_range, y_best, label="Función aproximada", color='green', linestyle='--')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(os.path.join(output, f"Function_{date_str}.png"))
    plt.show(block=False)

# plot tree
def plot_tree(individual, output, date_str):
    nodes, edges, labels = gp.graph(individual)

    dot = graphviz.Digraph(format='png')
    for node in nodes:
        label = str(labels.get(node, ""))
        dot.node(name=str(node), label=label)
        
    for (start, end) in edges:
        dot.edge(str(start), str(end))
    
    output_path = os.path.join(output, f"Tree_{date_str}")
    dot.render(output_path, view=True)