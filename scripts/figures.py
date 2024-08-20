import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def main():

    data = [
        [0.1, 0.14, 0.12, 0.08, 0.12, 0.12, 0.24, 0.14],
        [0.4, 0.48, 0.4, 0.46, 0.42, 0.38, 0.48, 0.58],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    #labels = [f'f = {l}' for l in [1.0, 0.5, 0.25, 0.1]]
    labels = ["$\\sigma = 0.0$", "$\\sigma = 0.001$", "$\\sigma = 0.1$"]
    #x_axis = ['0.0', '0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05', '0.1']
    x_axis = ['2', '4', '10', '20', '40', '100', '200', '400']
    x_label = "Number of Drift Events"
    y_label = "Detection Rate"

    fig, ax = plt.subplots(figsize=(10, 5))

    #ax.set_xscale("log")
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    #ax.minorticks_off()
    #ax.set_xticks(x_axis)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for i in range(len(data)):
        ax.plot(x_axis, data[i], label=labels[i], marker='x', linestyle=None)  #'--' if "ADWIN" in labels[i] else None)

    ax.legend()

    plt.savefig('incremental_high_eps.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
