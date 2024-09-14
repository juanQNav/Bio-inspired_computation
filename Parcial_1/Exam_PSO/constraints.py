def generic_constraint_example(position):
    return position[0] >= 0 and position[1] >= 0

#1 concurso de alambre (x = 5, y = 5)
def quadrilateral_perimeter_constraint(position):
    x, y = position[0], position[1]
    return 2 * (x + y)<= 20, abs((2 * (x + y))-20)

#2 cercar un solar rectangular (x = 100, y = 100)
def rectangle_perimeter_constraint(position):
    x, y = position
    return 2 * x + 2 * y <= 400, abs((2 * x + 2 * y) - 400)

#3 cercar un solar rectangular(otro) (x = 20, y = 20)
def rectangle_area_constraint(position):
    x, y = position[0], position[1]
    return x * y == 400 , abs((x * y) - 400)

#4 ventana rectangular (x = 1.54, y = 0.99)
def window_perimeter_constraint(position):
    x, y = position
    return 3 * x + 2 * y == 6.6, abs((3 * x + 2 * y) -6.6)

#5 jardin rectangular en terreno circular (x = 70.71, y = 70.71)
def circle_constraint(position):            
    x, y = position[0], position[1]
    return x**2 + y**2 == 100**2, abs((x**2 + y**2) - 100**2)

# ambitious thief
def knapsack_weight_constraint(position, weights, max_weight=10):
    total_weight = 0
    for i in range(len(position)):
        if position[i] > 0.5:  # The object is selected
            total_weight += weights[i]
    
    # Check if the total weight is within the allowable limit.
    if total_weight <= max_weight:
        return True, 0  # Complies with the restriction, no penalty
    else:
        return False, total_weight - max_weight

