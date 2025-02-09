import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
from .plotting import plot_errors

def load_knapsack_data(csv_path):
    weights = []
    values = []
    items_df = pd.read_csv(csv_path)
    for _, row in items_df.iterrows():
        weights.append(row['weights'])
        values.append(row['values'])

    return weights, values

def save_results(output_path, timestamp, best_position, errors, total_time, num_particles, max_epochs, dim, minx, maxx, w, c1, c2, error_function, positions_over_time, total_value=None):
    plot_file = os.path.join(output_path, f"pso_plot_{timestamp}.png")
    csv_file = os.path.join(output_path, f"pso_results_{timestamp}.csv")
    plot_trajectories_file = os.path.join(output_path, f'trajectories_plot_{timestamp}.png')

    # save plot_errors
    plot_errors(errors, w, c1, c2)
    plt.savefig(plot_file, bbox_inches='tight', pad_inches=0)

    # Save CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Number of particles", num_particles])
        writer.writerow(["Max Epochs", max_epochs])
        writer.writerow(["Dimensions", dim])
        writer.writerow(["Min x", minx])
        writer.writerow(["Max x", maxx])
        writer.writerow(["Inertia", w])
        writer.writerow(["C1", c1])
        writer.writerow(["C2", c2])
        writer.writerow(["Best Solution"] + list(map(str, best_position)))
        if total_value is not None:
            writer.writerow(["Total gain", total_value])
        writer.writerow(["Best Error", error_function(best_position)])
        writer.writerow(["Total Time (seconds)", total_time])
