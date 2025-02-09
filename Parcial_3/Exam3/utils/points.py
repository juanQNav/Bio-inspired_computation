import random

def gen_random_points(func, n_points=200, x_range=(-100, 100)):
    points = []
    for _ in range(n_points):
        x = random.uniform(*x_range)
        y = func(x)
        points.append((x, y))
    return points

def split_data(points, train_ratio=0.7):
    random.shuffle(points)
    split_idx = int(len(points) * train_ratio)
    return points[:split_idx], points[split_idx:]


