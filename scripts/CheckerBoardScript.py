import os

import numpy as np
import pandas as pd

import math

import random
import matplotlib.pyplot as plt

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

from src.CheckerBoardDetector import CheckerBoard1DUniform

F = 0.5
T = 2000
OFFSET = 0.0


def main():

    examples = pd.DataFrame([
        {0: 0.0, 't': 0},
        {0: 0.25, 't': 2500},
        {0: 0.4, 't': 5000},
        {0: 0.6, 't': 0},
        {0: 0.75, 't': 3000},
        {0: 0.8, 't': 5440},
    ])

    model = CheckerBoard1DUniform(F, T, offset=OFFSET)

    data = {i: {0: random.random() * 2.0 - 1.0, 't': i} for i in range(10000)}

    print(model.predict_many(examples))

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    x3 = []
    y3 = []
    x4 = []
    y4 = []

    for i in data:
        if random.random() < 0.05:
            if model.predict_one(data[i]) == 1:
                x1.append(i)
                y1.append(data[i][0])
            else:
                x2.append(i)
                y2.append(data[i][0])

            flag = (i // T) % 2 == 0
            val = ((data[i][0] + OFFSET) // F) % 2 == 0
            if flag:
                if val:
                    x3.append(i)
                    y3.append(data[i][0])
                else:
                    x4.append(i)
                    y4.append(data[i][0])
            else:
                if val:
                    x4.append(i)
                    y4.append(data[i][0])
                else:
                    x3.append(i)
                    y3.append(data[i][0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Example of Checkerboard Detection Method')
    ax.scatter(x1, y1, label='Class 1', facecolor="blue", hatch=10*"/", edgecolor="black")
    ax.scatter(x2, y2, label='Class 0', facecolor="orange", edgecolor="black")
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Feature Value')
    ax.legend()

    vlines = [T * i for i in range(1, 10000 // T)]
    plt.vlines(x=vlines, ymin=-1.0, ymax=1.0, colors='black', ls='--', lw=1)

    hlines = np.linspace(-1.0 + F, 1.0 - F, num=int(2 // F) - 1)
    plt.hlines(y=hlines, xmin=0, xmax=10000, colors='black', ls='--', lw=1)

    plt.savefig('CheckerBoardDetection.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
