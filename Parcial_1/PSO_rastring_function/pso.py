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

def show_vector(vector):
    for i in range(len(vector)):
        print("\n", end="")
        if vector[i] >= 0.0:
            print(' ', end="")
        print("%.4f" % vector[i], end="") # forur decimals
        print(" ", end="")
    print("\n")

def error(position):
    err = 0.0
    for i in range(len(position)):
        xi = position[i]
        err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return err

class Particle:
  def __init__(self, dim, minx, maxx, seed):
    self.rnd = random.Random(seed)
    self.position = [0.0 for i in range(dim)]
    self.velocity = [0.0 for i in range(dim)]
    self.best_part_pos = [0.0 for i in range(dim)]

    for i in range(dim):
      self.position[i] = ((maxx - minx) *
        self.rnd.random() + minx)
      self.velocity[i] = ((maxx - minx) *
        self.rnd.random() + minx)

    self.error = error(self.position) # curr error
    self.best_part_pos = copy.copy(self.position) 
    self.best_part_err = self.error # best error

def Solve(max_epochs, n, dim, minx, maxx, w, c1, c2):
  rnd = random.Random(0)

  # create n random particles
  swarm = [Particle(dim, minx, maxx, i) for i in range(n)] 

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
    
        swarm[i].velocity[k] = ( (w * swarm[i].velocity[k]) +
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
      swarm[i].error = error(swarm[i].position)

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
#end solve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--np", "--np",type=int, required=True, help="Number of particles")
    ap.add_argument("--mxe", "--mxe",type=int, required=True, help="Max Epochs")
    ap.add_argument("--dim", "--dim",type=int, required=True, help="Dimensions")
    ap.add_argument("--minx", "--minx",type=float, required=True, help="Min x")
    ap.add_argument("--maxx", "--maxx",type=float, required=True, help="Max x")
    ap.add_argument("--w", "--w",type=float, required=True, help="Inertia")
    ap.add_argument("--c1", "--c1",type=float, required=True, help="C1")
    ap.add_argument("--c2", "--c2",type=float, required=True, help="C2")
    ap.add_argument("--output", "--output",type=str, required=True, help="Output path")


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

    os.makedirs(output_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(output_path, f"pso_plot_{timestamp}.png")
    csv_file = os.path.join(output_path, f"pso_results_{timestamp}.csv")

    best_position, errors, total_time = Solve(max_epochs, num_particles, dim, minx, maxx, w, c1, c2)
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
        writer.writerow(["Best Error", error(best_position)])
        writer.writerow(["Total Time (seconds)", total_time])

    print("\nPSO completed\n")
    print("\nBest solution found:")
    show_vector(best_position)
    err = error(best_position)
    print("Error of best solution = %.6f" % err)