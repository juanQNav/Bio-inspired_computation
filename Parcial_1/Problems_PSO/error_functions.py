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