import os
import pandas as pd

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

import matplotlib.pyplot as plt
import numpy as np

from src.AgentStream import AgentStream, DataAgent, DataComponent
from src.CheckerBoardDetector import CheckerBoard1DUniform
import src.Analysis as analysis


SEED = None
T = 5000
SPLIT = 1000


def main():

    predictor = CheckerBoard1DUniform(0.5, SPLIT, flip=0.0, seed=SEED)
    model = AgentStream(predictor, seed=SEED)

    # Add Agent
    model.environment.add_agent(DataAgent('a', model, 0, x=.75, sigma=.25, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('b', model, 1, x=.25, sigma=.25, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('c', model, 1, x=.75, sigma=.25, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('d', model, 0, x=.25, sigma=.25, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))

    model.execute(T)

    print(f"Stream A weight: {model.environment.get_agent('a')[DataComponent].weight}")
    print(f"Stream B weight: {model.environment.get_agent('b')[DataComponent].weight}")
    print(f"Stream C weight: {model.environment.get_agent('c')[DataComponent].weight}")
    print(f"Stream D weight: {model.environment.get_agent('d')[DataComponent].weight}")

    records_d1_positive = np.array([x[0][0] for x in model.get_stream()
                                    if x[1] == 1 and 0 <= x[3] < 100])
    records_d2_positive = np.array([x[0][0] for x in model.get_stream()
                                    if x[1] == 1 and 900 <= x[3] < 1000])

    print(len(records_d2_positive) - len(records_d1_positive))

    records_d1_negative = np.array([x[0][0] for x in model.get_stream()
                                    if x[1] != 1 and 0 <= x[3] < 100])
    records_d2_negative = np.array([x[0][0] for x in model.get_stream()
                                    if x[1] != 1 and 900 <= x[3] < 1000])

    l_d1_pos = len(records_d1_positive)
    l_d2_pos = len(records_d2_positive)

    l_d1_neg = len(records_d1_negative)
    l_d2_neg = len(records_d2_negative)

    print(f'Lengths: d1 +: {l_d1_pos}, d1 -: {l_d1_neg}, d2 +: {l_d2_pos}, d2 -: {l_d2_neg}')

    _, kde1, mean1 = analysis.make_rescaled_distribution(
        records_d1_positive[:, np.newaxis], l_d1_pos, l_d2_pos, bandwidth=0.1)
    _, kde2, mean2 = analysis.make_rescaled_distribution(
        records_d2_positive[:, np.newaxis], l_d2_pos, l_d2_pos, bandwidth=0.1)

    _, kde3, mean3 = analysis.make_rescaled_distribution(
        records_d1_negative[:, np.newaxis], l_d1_neg, l_d2_neg, bandwidth=0.1)
    _, kde4, mean4 = analysis.make_rescaled_distribution(
        records_d2_negative[:, np.newaxis], l_d2_neg, l_d2_neg, bandwidth=0.1)

    fig, ax = plt.subplots()

    # KDE
    y1 = analysis.shape_data_to_rescaled_kde(records_d1_positive, kde1, mean1, l_d1_pos, l_d2_pos)
    y2 = analysis.shape_data_to_rescaled_kde(records_d1_positive, kde2, mean2, l_d2_pos, l_d2_pos)
    #print(analysis.test_distributions(y1, y2))
    #ax.plot(x, y1)
    #ax.plot(x, y2)
    #ax.plot(x, y2 - y1)
    ax.scatter(records_d1_positive, y1, label='D1 POS')
    ax.scatter(records_d1_positive, y2, label='D2 POS')
    delta_pos = y2 - y1
    ax.scatter(records_d1_positive, delta_pos, label='D2 - D1 POS')

    y1 = analysis.shape_data_to_rescaled_kde(records_d1_negative, kde3, mean3, l_d1_neg, l_d2_neg)
    y2 = analysis.shape_data_to_rescaled_kde(records_d1_negative, kde4, mean4, l_d2_neg, l_d2_neg)
    #print(analysis.test_distributions(y1, y2))
    ax.scatter(records_d1_negative, y1, label='D1 NEG')
    ax.scatter(records_d1_negative, y2, label='D2 NEG')
    delta_neg = y2 - y1
    ax.scatter(records_d1_negative, delta_neg, label='D2 - D1 NEG')

    print('Delta Tests')
    print(np.median(delta_pos[(0.1 < records_d1_positive) & (records_d1_positive < 0.4)]))
    print(np.median(delta_pos[(0.9 > records_d1_positive) & (records_d1_positive > 0.6)]))
    print(analysis.test_distributions(
        delta_pos[(0.1 < records_d1_positive) & (records_d1_positive < 0.4)],
        delta_pos[(0.9 > records_d1_positive) & (records_d1_positive > 0.6)])
    )
    #print(analysis.test_distributions(
    #    delta_pos[(0.9 > records_d1_positive) & (records_d1_positive > 0.6)],
    #    delta_neg[(0.9 > records_d1_negative) & (records_d1_negative > 0.6)])
    #)
    #print(analysis.test_distributions(delta_pos, delta_neg))

    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
