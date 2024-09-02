import random
import math
import copy
import sys
import time
import matplotlib.pyplot as plt
import argparse
import os
import csv
from datetime import datetime
import importlib

def load_error_function(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def load_constraints(module_name, function_names):
    module = importlib.import_module(module_name)
    return [getattr(module, func_name) for func_name in function_names]

def show_vector(vector):
    print(" ".join(f"{x: .4f}" for x in vector))

class Particle:
    def __init__(self, dim, minx, maxx, seed, error_function, constraints):
        self.rnd = random.Random(seed)
        self.position = [0.0 for i in range(dim)]
        self.velocity = [0.0 for i in range(dim)]
        self.best_part_pos = [0.0 for i in range(dim)]
        self.error_function = error_function
        self.constraints = constraints

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx) * self.rnd.random() + minx)

        # Calculate intial error
        self.error = self.evaluate(self.position)
        self.best_part_pos = copy.copy(self.position)
        self.best_part_err = self.error

    def evaluate(self, position):
        penalty = 0.0
        for constraint in self.constraints:
            check_const, dif = constraint(position)
            if not check_const:
                penalty += 1000 * dif  
        return self.error_function(position) + penalty


def Solve(max_epochs, n, dim, minx, maxx, w, c1, c2, error_function, constraints):
  rnd = random.Random(0)

  # create n random particles
  swarm = [Particle(dim, minx, maxx, i, error_function, constraints) for i in range(n)] 

  best_swarm_pos = [0.0 for i in range(dim)] # not necess.
  best_swarm_err = sys.float_info.max # swarm best
  for i in range(n): # check each particle
    if swarm[i].error < best_swarm_err:
      best_swarm_err = swarm[i].error
      best_swarm_pos = copy.copy(swarm[i].position) 

  epoch = 0
  errors = []
  start_time = time.time()

  while epoch < max_epochs:
    
    if epoch % 10 == 0 and epoch > 1:
      print("Epoch = " + str(epoch) +
        " best error = %.3f" % best_swarm_err)

    for i in range(n): # process each particle
      
      # compute new velocity of curr particle
      for k in range(dim): 
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()
    
        swarm[i].velocity[k] = ((w * swarm[i].velocity[k]) +
                                (c1 * r1 * (swarm[i].best_part_pos[k] -
                                swarm[i].position[k])) +  
                                (c2 * r2 * (best_swarm_pos[k] -
                                swarm[i].position[k])) )  

        if swarm[i].velocity[k] < minx:
          swarm[i].velocity[k] = minx
        elif swarm[i].velocity[k] > maxx:
          swarm[i].velocity[k] = maxx

      # compute new position using new velocity
      for k in range(dim): 
        swarm[i].position[k] += swarm[i].velocity[k]
        
      # compute error of new position
      swarm[i].error = swarm[i].evaluate(swarm[i].position)

      # is new position a new best for the particle?
      if swarm[i].error < swarm[i].best_part_err:
        swarm[i].best_part_err = swarm[i].error
        swarm[i].best_part_pos = copy.copy(swarm[i].position)

      # is new position a new best overall?
      if swarm[i].error < best_swarm_err:
        best_swarm_err = swarm[i].error
        best_swarm_pos = copy.copy(swarm[i].position)

    errors.append(best_swarm_err)
    # for-each particle
    epoch += 1
  end_time = time.time()
  # while
  total_time = end_time - start_time
  return best_swarm_pos, errors, total_time

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", type=int, required=True, help="Number of particles")
    ap.add_argument("--mxe", type=int, required=True, help="Max Epochs")
    ap.add_argument("--dim", type=int, required=True, help="Dimensions")
    ap.add_argument("--minx", type=float, required=True, help="Min x")
    ap.add_argument("--maxx", type=float, required=True, help="Max x")
    ap.add_argument("--w", type=float, required=True, help="Inertia")
    ap.add_argument("--c1", type=float, required=True, help="C1 cognitive (particle)")
    ap.add_argument("--c2", type=float, required=True, help="C2 social (swarw)")
    ap.add_argument("--output", type=str, required=True, help="Output path")
    ap.add_argument("--function", type=str, required=True, help="Error function name")
    ap.add_argument("--constraints", nargs='+', required=True, help="List of constraints")

    args = vars(ap.parse_args())

    num_particles = args["np"]
    max_epochs = args["mxe"]
    dim = args["dim"]
    minx = args["minx"]
    maxx = args["maxx"]
    w = args["w"]
    c1 = args["c1"]
    c2 = args["c2"]
    output_path = args["output"]
    function_name = args["function"]
    constraint_names = args["constraints"]

    error_function = load_error_function("error_functions", function_name)
    constraints = load_constraints("constraints", constraint_names)

    os.makedirs(output_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_path, f"pso_plot_{timestamp}.png")
    csv_file = os.path.join(output_path, f"pso_results_{timestamp}.csv")

    best_position, errors, total_time = Solve(max_epochs, num_particles, dim, minx, maxx, w, c1, c2, error_function, constraints)
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(f'PSO w={w}, c1 = {c1}, c2={c2}')
    # plt.show()
    plt.savefig(plot_file,bbox_inches='tight', pad_inches=0)

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
        writer.writerow(["Best Solution"] + best_position)
        writer.writerow(["Best Error", error_function(best_position)])
        writer.writerow(["Total Time (seconds)", total_time])

    print("\nPSO completed\n")
    print("\nBest solution found:")
    show_vector(best_position)
    err = min(errors)
    print("Error of best solution = %.6f" % err)