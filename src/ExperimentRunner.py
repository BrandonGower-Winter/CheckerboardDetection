import numpy as np
import json

from AgentStream import AgentStream, DataAgent, DataComponent, IntrinsicDriftSystem
from CheckerBoardDetector import CheckerBoard1DUniform
import Classifiers
import Analysis as analysis


def create_default_model(predictor, seed):
    model = AgentStream(predictor, seed=seed)
    # Add Agent
    model.environment.add_agent(DataAgent('a', model, 0, x=.75, sigma=.05, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('b', model, 1, x=.25, sigma=.05, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('c', model, 1, x=.75, sigma=.05, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    model.environment.add_agent(DataAgent('d', model, 0, x=.25, sigma=.05, weight_delta=.001,
                                          move_delta=0.0, feedback_type=[0, 0, 0, 0]))
    return model


class ModelBuilder:

    def __init__(self, filename: str):
        with open(filename) as f:  # TODO: This should probably be checked more thoroughly
            self.agents = json.load(f)

    def __call__(self, predictor, seed):
        model = AgentStream(predictor, seed=seed)

        for agent in self.agents:
            if agent['type'] == 'gauss':
                model.environment.add_agent(DataAgent(
                    agent['id'], model, agent['label'], agent['x_pos'], agent['y_pos'],
                    agent['weight'], agent['sigma'], agent['weight_delta'], agent['move_delta'],
                    agent['x_drift'], agent['y_drift'], agent['feedback_type'],
                    decay_feedback=agent['decay_feedback'] if 'decay_feedback' in agent else False
                ))
            elif agent['type'] == 'multi_gauss':
                centroids = np.linspace(agent['lower'], agent['upper'], agent['n'])
                if agent['weights'] == 'random':
                    weights = [model.random.random() for _ in range(len(centroids))]
                else:
                    weights = agent['weights']
                for i in range(len(centroids)):
                    model.environment.add_agent(DataAgent(  # TODO: Fix for multi-feature datasets
                        f'{agent["id"]}_{i}', model, agent['label'], centroids[i], 0.0,
                        weights[i], agent['sigma'], agent['weight_delta'], agent['move_delta'],
                        agent['x_drift'], 0.0, agent['feedback_type'],
                        decay_feedback=agent['decay_feedback'] if 'decay_feedback' in agent else False
                    ))
        return model


def load_model_configuration(filename: str):
    return ModelBuilder(filename)


# TODO: Need for each feature as well
def run_experiment(f: float, t: int, tau: int, w1: int, model_builder: callable = None,
                   warmup: int = 5, seed: int = None, n_classes: int = 2, offset: float = 0.0,
                   classifier: str = None, mix: float = 0.0,  # TODO, allowing specifying of class label names
                   itype: str = 'none', imag: float = 1.0, ifreq: int = 1, ireset: int = -1):

    if classifier is None:
        predictor = CheckerBoard1DUniform(f, tau, seed=seed)
    elif classifier == 'random':
        predictor = Classifiers.Mixer(mix, CheckerBoard1DUniform(f, tau, seed=seed),
                                      Classifiers.RandomClassifier(2, seed), seed)
    else:
        predictor = Classifiers.Mixer(mix, CheckerBoard1DUniform(f, tau, seed=seed),
                                      Classifiers.ThresholdClassifier(), seed)

    model = model_builder(predictor, seed) if model_builder is not None else create_default_model(predictor, seed)

    if itype == 'sudden':
        model.systems.add_system(IntrinsicDriftSystem("isystem", model, frequency=ifreq,
                                                      drift_type=IntrinsicDriftSystem.DRIFT_SUDDEN))
    elif itype == "gradual":
        model.systems.add_system(IntrinsicDriftSystem("isystem", model, ireset, imag, frequency=ifreq,
                                                      drift_type=IntrinsicDriftSystem.DRIFT_GRADUAL,
                                                      drift_mode=IntrinsicDriftSystem.DRIFT_INDIVIDUAL))
    #print(model)

    RESULTS = {}
    for i in range(n_classes):
        RESULTS[i] = {'a': [], 'b': [], 'p': []}

    for i in range(t // tau):
        trial_frame = {}
        model.execute(tau)

        start_i = tau * i
        end_i = tau * (i + 1)

        # Log the Stream Weights
        trial_frame['timestep'] = end_i
        stream_data = {}
        for agent in model.environment.get_agents(DataComponent):
            stream_data[agent.id] = agent[DataComponent].weight

        trial_frame['stream_data'] = stream_data

        # Used to determine which blocks are predicting all positives / or all negatives
        T_MASK = i % 2 == 0

        # For Each Class
        for c in range(n_classes):  # TODO: Make it multi-class

            records_d1 = np.array([x[0][0] for x in model.get_stream()
                                        if x[1] == c and start_i <= x[3] < start_i + w1])
            records_d2 = np.array([x[0][0] for x in model.get_stream()
                                        if x[1] == c and end_i - w1 <= x[3] < end_i])

            l_d1 = len(records_d1)
            l_d2 = len(records_d2)

            # Get Data based on feature length split
            # Get index = floor((d1 + offset) // f) % 2
            mask = np.floor((records_d1 + offset) // f) % 2 == 0
            med1 = len(records_d1[mask]) / l_d1 if l_d1 > 0 else 0
            med2 = len(records_d1[~mask]) / l_d1 if l_d1 > 0 else 0

            mask = np.floor((records_d2 + offset) // f) % 2 == 0
            med1 = len(records_d2[mask]) / l_d2 - med1 if l_d2 > 0 else -med1
            med2 = len(records_d2[~mask]) / l_d2 - med2 if l_d2 > 0 else -med2

            if T_MASK:
                RESULTS[c]['a'].append(med1)
                RESULTS[c]['b'].append(med2)
            else:
                RESULTS[c]['b'].append(med1)
                RESULTS[c]['a'].append(med2)

            # If cutoff, perform test
            if len(RESULTS[c]['a']) > warmup:
                RESULTS[c]['p'].append(analysis.test_distributions(RESULTS[c]['a'], RESULTS[c]['b']).pvalue)

    print(model)
    return RESULTS
