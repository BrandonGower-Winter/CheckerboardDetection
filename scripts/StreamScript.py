import os

from ECAgent.Environments import PositionComponent

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

import matplotlib.pyplot as plt
import numpy as np
from src.AgentStream import AgentStream, DataAgent

SEED = 123456
T = 1000


def predictor(x, y, t):
    if x[0] > 1.0 and x[1] > 1.0:
        return 1
    else:
        return 0


def main():
    model = AgentStream(predictor, seed=SEED, drift_magnitude=0.1, drift_reset=100, drift_mode=1)

    # Add Agent
    model.environment.add_agent(DataAgent('a', model, 1, x=-5, weight=1, weight_delta=0.0,
                                          move_delta=.1, feedback_type=[1, 0, 0, 0]))
    model.environment.add_agent(DataAgent('b', model, 1, weight_delta=1.0,
                                          move_delta=.1, feedback_type=[1, 0, 0, 0]))

    agent_apos = ([], [])
    agent_bpos = ([], [])

    for _ in range(T):
        model.execute()
        agent_apos[0].append(model.environment.agents['a'][PositionComponent].x)
        agent_apos[1].append(model.environment.agents['a'][PositionComponent].y)

        agent_bpos[0].append(model.environment.agents['b'][PositionComponent].x)
        agent_bpos[1].append(model.environment.agents['b'][PositionComponent].y)

    data = list(zip(*[instance[0] for instance in model.get_stream()]))
    plt.scatter(data[0], data[1])
    plt.scatter(agent_apos[0], agent_apos[1])
    plt.scatter(agent_bpos[0], agent_bpos[1])
    plt.show()


if __name__ == '__main__':
    main()
