import os

import numpy as np
import pandas as pd

import math

import random
import matplotlib.pyplot as plt

from Classifiers import RandomClassifier, ThresholdClassifier
from ExperimentRunner import ModelBuilder

FILE_PATH = "./configs/self_fulfilling/default_1000.json"


def main():
    mb = ModelBuilder(FILE_PATH)

    # For RC
    model = mb(ThresholdClassifier(), None)
    model.execute(10000)

    data = [(x[0][0], x[1], x[3]) for x in model.get_stream()]

    class_0 = [(x[0], x[2]) for x in data if x[1] == 0 and random.random() < 0.1]
    class_1 = [(x[0], x[2]) for x in data if x[1] == 1 and random.random() < 0.1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Visualization of PD induced by Threshold Classifier')
    ax.scatter(
    [x[1] for x in class_1],
    [x[0] for x in class_1],
    label='Class 1', facecolor="blue", hatch=10 * "/", edgecolor="black")
    ax.scatter(
    [x[1] for x in class_0],
    [x[0] for x in class_0],
    label='Class 0', facecolor="orange", edgecolor="black")
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Feature Value')
    ax.legend()

    plt.savefig('TCViz.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
