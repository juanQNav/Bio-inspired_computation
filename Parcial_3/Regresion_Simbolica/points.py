import random

def gen_random_points(range):
    points = []
    for i in range:
        points.append(random.randint(-100, 100))
    return points


if __name__ == "__main__":
    print(gen_random_points(range(20)))
