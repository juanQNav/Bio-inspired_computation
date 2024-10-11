import math

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