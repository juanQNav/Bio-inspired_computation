import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_errors(best_fitness_per_gen, date_str, o):
    plt.plot(best_fitness_per_gen, label='Best Fitness (Error)')
    plt.title('Evolution of the Best Individual')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Error)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(o,f"fitness_evolution_{date_str}.png"))
    plt.show(block=False)