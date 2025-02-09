import math

# Protected division
def protectedDiv(left, right):
    if abs(right) < 1e-10: 
        return 1.0  
    try:
        return left / right
    except Exception:  
        return 1.0
# Protected square root
def protectedSqrt(x):
    return math.sqrt(x) if x >= 0 else 0.0

# Protected natural logarithm
def protectedLog(x):
    return math.log(x) if x > 0 else 0.0

# Exponential protected    
def protectedExp(x):
    try:
        return math.exp(x) if x < 700 else 1e10  # Limits the maximum value
    except OverflowError:
        return 1e10  # Safe value in case of error

# Tangent protected
def protectedTan(x):
    try:
        result = math.tan(x)
        return result if abs(result) < 1e10 else 0.0
    except ValueError:
        return 0.0
