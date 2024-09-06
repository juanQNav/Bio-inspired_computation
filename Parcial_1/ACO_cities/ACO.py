from __future__ import division
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt
from sko.ACA import ACA_TSP
import argparse
import pandas as pd
import os
from datetime import datetime
import csv
import time

def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=str, required=True, help="Source file csv")
    ap.add_argument("--a", type=int, required=True, help="Size of ants colony")
    ap.add_argument("--i", type=int, required=True, help="Iterations")
    ap.add_argument("--o", type=str, required=True, help="Output")

    args = vars(ap.parse_args())

    csv_dir = args["d"]
    size_colony = args["a"]
    iterations = args["i"]
    output_path = args['o']

    os.makedirs(output_path, exist_ok=True)

    points_csv = pd.read_csv(csv_dir, comment='#')

    points_coordinate = points_csv.to_numpy()
    num_points = len(points_coordinate)
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')

    print(f"iterations: {iterations}")
    print(points_csv)
    print(f'Number of points: {num_points}')


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_path, f"aco_plot_{timestamp}.png")
    csv_file = os.path.join(output_path, f"aco_results_{timestamp}.csv")

    start_time = time.time()

    aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
                  size_pop=size_colony, max_iter=iterations,
                  distance_matrix=distance_matrix)
    
    best_x, best_y = aca.run()

    end_time = time.time()
    total_time = end_time - start_time

    # Plot the result
    fig, ax = plt.subplots(1, 2)
    best_points_ = np.concatenate([best_x, [best_x[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]

    for index in range(0, len(best_points_)):
        ax[0].annotate(best_points_[index], (best_points_coordinate[index, 0], best_points_coordinate[index, 1]))
    ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
    pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
    plt.show(block=False)

    print(f"Final distance: {best_y}")
    ## outputs
    plt.savefig(plot_file,bbox_inches='tight', pad_inches=0)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Colony Ants", size_colony])
        writer.writerow(["Iterations", iterations])
        writer.writerow(["Number of points", num_points])
        writer.writerow(["Final distance", best_y])

    input("Press enter to continue...")

# references: https://people.sc.fsu.edu/~jburkardt/datasets/cities/cities.html
# https://medium.com/@sakamoto2000.kim/ant-colony-optimization-aco-in-the-travel-salesman-problem-tsp-54f83ccd9eff