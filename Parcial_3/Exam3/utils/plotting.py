import numpy as np
import matplotlib.pyplot as plt
import os
from deap import gp
import graphviz

#plot function
def plot_function(train_points, test_points, best_func, label_func, output, date_str, figsize=(10, 6), view=True):
    plt.figure(figsize=figsize)
    plt.title(f"Function: {label_func}")

    x_train, y_train = zip(*train_points)
    x_test, y_test = zip(*test_points)

    plt.scatter(x_train, y_train, label="Train points", color="#FF5733")
    plt.scatter(x_test, y_test, label="Test points", color="#90EE90")

    x_pred = np.sort(x_test)
    y_pred = [best_func(x) for x in x_pred]

    plt.plot(x_pred, y_pred, label="Prediction line (Test)", color="blue", linestyle='--')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output, f"Function_{label_func}_{date_str}.png"))
    if view:
        plt.show(block=False)
    else:
        plt.close()

# plot tree
def plot_tree(individual, output, date_str, view=True):
    nodes, edges, labels = gp.graph(individual)

    dot = graphviz.Digraph(format='png')
    
    for node in nodes:
        label = str(labels.get(node, ""))
        dot.node(name=str(node), label=label)

    for (start, end) in edges:
        dot.edge(str(start), str(end))
    
    output_path = os.path.join(output, f"Tree_{date_str}")
    
    dot.render(output_path, view=view, cleanup=True)
