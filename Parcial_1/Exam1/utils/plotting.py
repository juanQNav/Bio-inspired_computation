import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_errors(errors,w,c1,c2):
    fig = plt.figure()
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title(f'PSO w={w}, c1 = {c1}, c2={c2}')
    return fig