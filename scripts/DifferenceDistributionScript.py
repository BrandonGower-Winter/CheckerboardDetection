import os
import pandas as pd

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

import matplotlib.pyplot as plt
import numpy as np

from src.AgentStream import AgentStream, DataAgent, DataComponent
from src.CheckerBoardDetector import CheckerBoard1DUniform
import src.Analysis as analysis
from sklearn.neighbors import KernelDensity

SEED = None
T = 100000
SPLIT = 1000
F_SPLIT = 1.0
WINDOW = 100

MIN_RESULTS = 0
MAX_RESULTS = 50

def main():
    predictor = CheckerBoard1DUniform(F_SPLIT, SPLIT, flip=0.0, seed=SEED)
    model = AgentStream(predictor, seed=SEED)

    # Add Agent
    #model.environment.add_agent(DataAgent('a', model, 0, x=-.5, sigma=.1, weight_delta=.005,
    #                                      move_delta=0.0, feedback_type=[1, 0, 0, 0]))
    model.environment.add_agent(DataAgent('b', model, 1, x=-.5, sigma=.05, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[1, 0, 0, 0], decay_feedback=True))
    model.environment.add_agent(DataAgent('c', model, 1, x=.5, sigma=.05, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[1, 0, 0, 0], decay_feedback=True))
    #model.environment.add_agent(DataAgent('d', model, 0, x=.5, sigma=.1, weight_delta=.005,
    #                                      move_delta=0.0, feedback_type=[1, 0, 0, 0]))

    GROUP_A = []
    GROUP_B = []

    for i in range(T // SPLIT):
        model.execute(SPLIT)

        start_i = SPLIT * i
        end_i = SPLIT * (i + 1)
        print(f"Timestep: {end_i}")
        #print(f"Stream A weight: {model.environment.get_agent('a')[DataComponent].weight}")
        print(f"Stream B weight: {model.environment.get_agent('b')[DataComponent].weight}")
        print(f"Stream C weight: {model.environment.get_agent('c')[DataComponent].weight}")
        #print(f"Stream D weight: {model.environment.get_agent('d')[DataComponent].weight}")

        records_d1_positive = np.array([x[0][0] for x in model.get_stream()
                                        if x[1] == 1 and start_i <= x[3] < start_i + WINDOW])
        records_d2_positive = np.array([x[0][0] for x in model.get_stream()
                                        if x[1] == 1 and end_i - WINDOW <= x[3] < end_i])

        l_d1_pos = len(records_d1_positive)
        l_d2_pos = len(records_d2_positive)

        #_, kde1, mean1 = analysis.make_rescaled_distribution(
        #    records_d1_positive[:, np.newaxis], l_d1_pos, l_d2_pos, bandwidth=0.1)
        #_, kde2, mean2 = analysis.make_rescaled_distribution(
        #    records_d2_positive[:, np.newaxis], l_d2_pos, l_d2_pos, bandwidth=0.1)

        # KDE
        #y1 = analysis.shape_data_to_rescaled_kde(records_d1_positive, kde1, mean1, l_d1_pos, l_d2_pos)
        #y2 = analysis.shape_data_to_rescaled_kde(records_d1_positive, kde2, mean2, l_d2_pos, l_d2_pos)
        #delta_pos = y2 - y1

        mask = np.abs(records_d1_positive // F_SPLIT) % 2 == 0
        med1 = len(records_d1_positive[mask]) / l_d1_pos
        med2 = len(records_d1_positive[~mask]) / l_d1_pos

        mask = np.abs(records_d2_positive // F_SPLIT) % 2 == 0
        med1 = len(records_d2_positive[mask]) / l_d2_pos - med1
        med2 = len(records_d2_positive[~mask]) / l_d2_pos - med2

        #med1 = np.median(med1) if len(med1) > 0 else 0.0
        #med2 = np.median(med2) if len(med2) > 0 else 0.0

        GROUP_A.append(med1)
        GROUP_B.append(med2)
        print(f'Deltas A: {GROUP_A[-1]} B: {GROUP_B[-1]}')

        # Plot Distribution Changes
        kde1 = KernelDensity(kernel='gaussian', bandwidth=0.1)
        kde1.fit(records_d1_positive[:, np.newaxis])

        kde2 = KernelDensity(kernel='gaussian', bandwidth=0.1)
        kde2.fit(records_d2_positive[:, np.newaxis])

        fig, ax = plt.subplots()
        X = np.linspace(-1.0, 1.0, 1000)[:, np.newaxis]
        ax.plot(X, np.exp(kde1.score_samples(X)), label='Start')
        ax.plot(X, np.exp(kde2.score_samples(X)), label='END')
        ax.legend()
        fig.savefig(f'distribution_{i}.png', dpi=fig.dpi)
        plt.close(fig)

    EVENS_A = [GROUP_A[i] for i in range(len(GROUP_A)) if i % 2 == 0]
    ODDS_A = [GROUP_A[i] for i in range(len(GROUP_A)) if i % 2 == 1]

    EVENS_B = [GROUP_B[i] for i in range(len(GROUP_B)) if i % 2 == 0]
    ODDS_B = [GROUP_B[i] for i in range(len(GROUP_B)) if i % 2 == 1]

    regime1 = [
        EVENS_A[i // 2] if i % 2 == 0 else ODDS_B[i // 2] for i in range(2*len(EVENS_A))
    ]
    regime2 = [
        ODDS_A[i // 2] if i % 2 == 0 else EVENS_B[i // 2] for i in range(2 * len(ODDS_A))
    ]
    print(analysis.test_distributions(regime1[MIN_RESULTS:(MIN_RESULTS + MAX_RESULTS)],
                                      regime2[MIN_RESULTS:(MIN_RESULTS + MAX_RESULTS)]))

    N1 = len(regime1)
    mv_avg_r1 = [np.convolve(regime1[0:x], np.ones(x)/x, mode='valid')[0] if x != 0 else 0 for x in range(N1)]

    N2 = len(regime2)
    mv_avg_r2 = [np.convolve(regime2[0:x], np.ones(x)/x, mode='valid')[0] if x != 0 else 0 for x in range(N2)]

    fig, ax = plt.subplots()
    ax.plot(np.arange(N1), regime1, label='R1')
    ax.plot(np.arange(N2), regime2, label='R2')
    ax.plot(np.arange(N1), mv_avg_r1, label='R1-mvavg')
    ax.plot(np.arange(N2), mv_avg_r2, label='R2-mvavg')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
