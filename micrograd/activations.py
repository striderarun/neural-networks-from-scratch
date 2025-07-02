import matplotlib.pyplot as plt
import numpy as np

# Smoothly squashes functions between -1 and 1. 
def plot_tanh_function():
    xs = np.arange(-5, 5, 0.5)
    ys = np.tanh(xs)
    print(xs)
    print(ys)
    plt.plot(xs, ys)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    plot_tanh_function()