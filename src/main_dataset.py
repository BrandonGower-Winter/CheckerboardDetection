import argparse
import json
import pandas as pd
import numpy as np
import random
import statistics

from CheckerBoardDetector import CheckerBoard1DUniform


LABEL = "label"
INSTANCE = 'id'
SEED = None

N = 1  # Repetitions
T = 100000  # Number of Instances
SIGMA = 0.001  # Influence Strength
F = 0.5
TAU = 1000  # Length of Trial
W1 = 100  # Window from beginning and end of trial to take instances from when constructing d1 and d2 distributions

OUTPUT = "./results.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Dataset to Import.', type=str)
    parser.add_argument('--label', help='Name of label in dataset', default=LABEL, type=str)
    parser.add_argument('--instance', help='Name of Instance ID in dataset', default=INSTANCE, type=str)

    parser.add_argument('-n', help='Number of simulations to run.', default=N, type=int)
    parser.add_argument('-t', help='Number of instances to generate', default=T, type=int)
    parser.add_argument('--sigma', help='Strength of the Performative Drift', default=SIGMA, type=float)
    parser.add_argument('--tau', help='The length of a trial period. Value MUST integer divide with T',
                        default=TAU, type=int)
    parser.add_argument('-f', help='Length of a Checkerboard in the feature space.', default=F, type=float)
    parser.add_argument('--wone', help='Window from beginning and end of trial to take instances '
                                       'from when constructing d1 and d2 distributions', default=W1, type=int)
    parser.add_argument('--seed', help="Specify the seed for the Model's pseudorandom number generator", default=None,
                        type=int)

    parser.add_argument('-o', '--output', help="The name of the output file to write results to.",
                        default=OUTPUT, type=str)

    parser.add_argument('--debug',
                        help='Sets the model to debug mode. Output printed to terminal will be verbose',
                        action='store_true')

    return parser.parse_args()


def run_experiment(instances, labels, args):
    # Initialize Centroids and Weights
    centroids = np.arange(instances.shape[0]).tolist()
    N = len(centroids)
    W = 1 / N
    weights = np.full(N, W).tolist()
    print(f"Generated {N} centroids with weights: {W}")

    # Create Predictor
    random.seed(args.seed)
    # This needed to make the sigma values equivalent to the 200 centroid experiments
    sigma = args.sigma  # args.sigma * len(centroids) / 200

    model = CheckerBoard1DUniform(args.f, args.tau, seed=args.seed)
    stream = []

    # Run Simulation
    for t in range(args.t):
        # Get weighted random instance
        i = random.choices(centroids, weights=weights)[0]
        x = instances[i]
        y_hat = statistics.mode([model.predict_one({0: x[j], 't': t}) for j in range(len(x))])

        if y_hat == labels[i]:
            weights[i] += sigma
        else:
            weights[i] = max(W, weights[i] - sigma)

        stream.append((labels[i], y_hat))

    RESULTS = {}
    for i in range(2):
        RESULTS[i] = {'a': [], 'b': []}

    # Run Tests
    for i in range(args.t // args.tau):
        trial_frame = {}

        start_i = args.tau * i
        end_i = args.tau * (i + 1)

        # Log the Stream Weights
        trial_frame['timestep'] = end_i

        # For Each Class
        for c in range(2):

            records_d1 = np.array([stream[t] for t in range(start_i, start_i + args.wone) if stream[t][0] == c])
            records_d2 = np.array([stream[t] for t in range(end_i - args.wone, end_i) if stream[t][0] == c])

            l_d1 = len(records_d1)
            l_d2 = len(records_d2)

            # Get Data based on feature length split
            mask = len([x[1] for x in records_d1 if x[1] == 0])
            med1 = mask / l_d1 if l_d1 > 0 else 0
            med2 = (l_d1 - mask) / l_d1 if l_d1 > 0 else 0

            mask = len([x[1] for x in records_d2 if x[1] == 0])
            med1 = mask / l_d2 - med1 if l_d2 > 0 else -med1
            med2 = (l_d2 - mask) / l_d2 - med2 if l_d2 > 0 else -med2

            RESULTS[c]['a'].append(med1)
            RESULTS[c]['b'].append(med2)

    return RESULTS


def write_to_file(filename, data):
    # Serializing json
    json_object = json.dumps(data, indent=4)

    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def main():
    args = parse_args()

    df = pd.read_csv(args.d)
    print(df.head())

    labels = df[args.label].values

    del df[args.instance]
    del df[args.label]

    instances = df.values

    if args.debug:
        np.seterr(invalid='raise')

    data = []
    for i in range(args.n):
        print(f"Running Experiment: {i}...")
        data.append(run_experiment(instances, labels, args))
        print("Complete!")

    print(f"Writing Results to {args.output}...")
    write_to_file(args.output, data)
    print("Done!")


if __name__ == '__main__':
    main()
