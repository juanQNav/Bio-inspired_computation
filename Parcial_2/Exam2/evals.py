import math
import numpy as np


def eval_rastrigin(individual):
    err = 0.0
    for i in range(len(individual)):
        xi = individual[i]
        err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return (err,)


def eval_knapsack(individual, weights, values, max_weight=10):
    total_value = 0
    total_weight = 0

    for i in range(len(individual)):
        if individual[i] > 0.5:  # The object is selected
            total_value += values[i]
            total_weight += weights[i]

    # If the weight exceeds the maximum allowed, apply a penalty.
    if total_weight > max_weight:
        return (0,)
    return (-total_value,)


def eval_classification(individual, data, target):
    """
    Evalúa un individuo para el problema de clasificación.
    Los parámetros del individuo son (A, B, C, D, E, F) que se usan en la ecuación de clasificación.

    Parámetros:
    - individual: List[float] -> parámetros (A, B, C, D, E, F) del individuo.
    - data: np.array -> Datos de entrada del Iris dataset (PetalLength, PetalWidth).
    - target: np.array -> Objetivos de clase, donde Setosa=0, Versicolor=1, Virginica=2.

    Retorno:
    - Tuple[float]: Error total del individuo.
    """
    A, B, C, D, E, F = individual

    # Inicializar el error acumulado
    total_error = 0.0

    # Recorrer cada ejemplo en el dataset
    for i in range(len(data)):
        petal_length = data[i][2]  # Índice 2 es PetalLength
        petal_width = data[i][3]   # Índice 3 es PetalWidth

        # Calcular el valor de la función f(PetalLength, PetalWidth)
        try:
            f_value = ((A / B) * petal_length) + \
                ((C / D) * petal_width) + (E / F)
        except ZeroDivisionError:
            # Penalizar fuertemente si hay división por cero
            return (float('inf'),)

        # Obtener la clase objetivo
        target_class = target[i]

        # Calcular el error respecto al valor objetivo 0, 1, o 2
        error = abs(f_value - target_class)

        # Acumular el error total
        total_error += error

    # Retornar el error total como una tupla
    return (total_error,)
