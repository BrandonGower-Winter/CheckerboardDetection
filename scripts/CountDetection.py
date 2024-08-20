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
T = 100000
SPLIT = 1000
WINDOW = 300

def main():
    predictor = CheckerBoard1DUniform(0.5, SPLIT, flip=0.0, seed=SEED)
    model = AgentStream(predictor, seed=SEED)

    # Add Agent
    model.environment.add_agent(DataAgent('a', model, 0, x=.75, sigma=.1, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('b', model, 1, x=.25, sigma=.1, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('c', model, 1, x=.75, sigma=.1, weight_delta=.001, weight=2.0,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('d', model, 0, x=.25, sigma=.1, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))

    C1 = []
    C2 = []

    for i in range(T // SPLIT):
        model.execute(SPLIT)

        start_i = SPLIT * i
        end_i = SPLIT * (i + 1)
        print(f"Timestep: {end_i}")
        print(f"Stream A weight: {model.environment.get_agent('a')[DataComponent].weight}")
        print(f"Stream B weight: {model.environment.get_agent('b')[DataComponent].weight}")
        print(f"Stream C weight: {model.environment.get_agent('c')[DataComponent].weight}")
        print(f"Stream D weight: {model.environment.get_agent('d')[DataComponent].weight}")

        records_d1 = np.array([x[0][0] for x in model.get_stream()
                                    if x[1] == 1 and start_i <= x[3] < start_i + WINDOW])
        records_d1_correct = np.array([x[1] == x[2] for x in model.get_stream()
                               if x[1] == 1 and start_i <= x[3] < start_i + WINDOW])

        records_d2 = np.array([x[0][0] for x in model.get_stream()
                                    if x[1] == 1 and end_i - WINDOW <= x[3] < end_i])
        records_d2_correct = np.array([x[1] == x[2] for x in model.get_stream()
                               if x[1] == 1 and end_i - WINDOW <= x[3] < end_i])

        d2_mask = records_d2 < 0.5
        END = records_d2

        END_CORRECT_C1 = END[d2_mask][records_d2_correct[d2_mask]]
        END_CORRECT_C2 = END[~d2_mask][records_d2_correct[~d2_mask]]
        d1_mask = records_d1 < 0.5
        START = records_d1
        START_CORRECT = START[d1_mask][records_d1_correct[d1_mask]]

        LEND = len(END)
        LEND_CORRECT_C1 = len(END_CORRECT_C1)
        LEND_CORRECT_C2 = len(END_CORRECT_C2)

        LSTART = len(START)
        LSTART_CORRECT = len(START_CORRECT)
        C1.append(LEND_CORRECT_C1 / LEND)
        C2.append(LEND_CORRECT_C2 / LEND)
        print(f"Change in Class 1 (C1): {C1[-1]}")
        print(f"Change in Class 1 (C2): {C2[-1]}")
        model.clear_stream()

    EVENS_C1 = [C1[i] for i in range(len(C1)) if i % 2 == 0]
    ODDS_C1 = [C1[i] for i in range(len(C1)) if i % 2 == 1]

    EVENS_C2 = [C2[i] for i in range(len(C2)) if i % 2 == 0]
    ODDS_C2 = [C2[i] for i in range(len(C2)) if i % 2 == 1]

    print(analysis.test_distributions(EVENS_C1, ODDS_C2))
    print(analysis.test_distributions(ODDS_C1, EVENS_C2))

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(EVENS_C1)), EVENS_C1, label='EC1')
    ax.plot(np.arange(len(EVENS_C2)), EVENS_C2, label='EC2')
    ax.plot(np.arange(len(ODDS_C1)), ODDS_C1, label="OC1")
    ax.plot(np.arange(len(ODDS_C2)), ODDS_C2, label="OC2")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
