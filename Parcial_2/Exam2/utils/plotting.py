import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score


def plot_errors(best_fitness_per_gen, date_str, o):
    plt.plot(best_fitness_per_gen, label='Best Fitness (Error)')
    plt.title('Evolution of the Best Individual')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Error)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(o, f"fitness_evolution_{date_str}.png"))
    plt.show(block=False)


def plot_classification_scatter(best_params, data, target, date_str, o):
    def predict_class(petal_length, petal_width, params):
        A, B, C, D, E, F = params
        f_value = ((A / B) * petal_length) + ((C / D) * petal_width) + (E / F)
        if f_value < 0.5:
            return 0
        elif f_value < 1.5:
            return 1
        else:
            return 2

    predictions = [predict_class(petal_length, petal_width, best_params)
                   for petal_length, petal_width in data[:, 2:4]]

    accuracy = accuracy_score(target, predictions)
    error_rate = (1 - accuracy) * 100

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Error rate: {error_rate:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 2], data[:, 3], c=target,
                cmap='viridis', marker='o', label='True class')
    plt.scatter(data[:, 2], data[:, 3], c=predictions,
                cmap='coolwarm', marker='x', label='Predicted class', alpha=0.7)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend(["True Class", "Predicted Class"])
    plt.colorbar(label='Clase')
    plt.title('Classification Scatter Plot')
    plt.savefig(os.path.join(o, f"classification_scatter_plot_{date_str}.png"))
    plt.show(block=False)
