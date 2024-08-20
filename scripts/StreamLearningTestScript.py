import os

from river import metrics
from river.naive_bayes import GaussianNB

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

from src.AgentStream import AgentStream, DataAgent

SEED = 123456
T = 1000


def main():
    model = GaussianNB()
    accuracy = metrics.Accuracy()

    def predictor(x, y, t):
        x_i = {0: x[0], 1: x[1]}
        y_hat = model.predict_one(x_i)
        if y_hat is not None:
            accuracy.update(y, y_hat)

        # Learn
        model.learn_one(x_i, y)

        return y_hat

    simulator = AgentStream(predictor, seed=SEED, drift_magnitude=0.1, drift_reset=0, drift_mode=1)

    # Add Agent
    simulator.environment.add_agent(DataAgent('a', simulator, 1, x=-1, weight=1, weight_delta=0.0,
                                          move_delta=.1, feedback_type=[0, 0, 0, 1]))
    simulator.environment.add_agent(DataAgent('b', simulator, 0, weight_delta=0.0,
                                          move_delta=.0, feedback_type=[0, 0, 0, 0]))

    for _ in range(T):
        simulator.execute()
        print(f'Model Accuracy: {accuracy}')


if __name__ == '__main__':
    main()
