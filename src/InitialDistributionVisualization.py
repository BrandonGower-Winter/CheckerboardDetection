import os

import numpy as np
import pandas as pd

import math

import random
import matplotlib.pyplot as plt

from Classifiers import RandomClassifier
from ExperimentRunner import ModelBuilder
from sklearn.neighbors import KernelDensity

FILE_PATH = "./configs/self_fulfilling/default.json"


def main():
    mb = ModelBuilder(FILE_PATH)

    # For RC
    model = mb(RandomClassifier(), None)
    model.execute(1000)

    data = [(x[0][0], x[1], x[3]) for x in model.get_stream()]

    class_0 = np.array([x[0] for x in data if x[1] == 0])
    class_1 = np.array([x[0] for x in data if x[1] == 1])

    # Plot Distribution Changes
    kde1 = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kde1.fit(class_0[:, np.newaxis])

    kde2 = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kde2.fit(class_1[:, np.newaxis])

    fig, ax = plt.subplots(figsize=(10, 5))
    X = np.linspace(-1.0, 1.0, 1000)[:, np.newaxis]
    ax.fill_between(X[:, 0], np.exp(kde1.score_samples(X)), label='Class 0', alpha=0.2)
    ax.fill_between(X[:, 0], np.exp(kde2.score_samples(X)), label='Class 1', alpha=0.2)
    ax.set_ylabel('Density')
    ax.set_xlabel('Feature Value')
    ax.legend()
    plt.savefig('distribution_3.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
