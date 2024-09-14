import math

def error(position):
    err = 0.0
    for i in range(len(position)):
        xi = position[i]
        err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return err

# 1 2 5concurso de alambre
def fun_max_area_quadrilateral(position):
    x, y = position
    area = x * y
    return -area  # We deny the area to convert the problem to minimization.

#3 cercar un solar rectangular (otro)
def fun_min_cost_valla(position):
    x, y = position
    cost = 8 * (x + y)
    return cost

#4 ventana rectangular
def fun_max_window_area(position):
    x,y = position
    area_rect = x * y
    area_tri = (math.sqrt(3) / 4) * (x ** 2)
    total_area = area_rect + area_tri
    return -total_area

# ambitious thief
def knapsack_error(position, weights, values, max_weight=10):
    total_value = 0
    total_weight = 0
    
    for i in range(len(position)):
        if position[i] > 0.5:  # The object is selected
            total_value += values[i]
            total_weight += weights[i]
    
    # If the weight exceeds the maximum allowed, apply a penalty.
    if total_weight > max_weight:
        return total_value
    
    return -total_value  # Negativo to minimize