import argparse
import numpy as np
import river.drift as drift

import Classifiers as classifers
import ExperimentRunner as runner

N = 1  # Repetitions
T = 100000  # Number of Instances
DETECTOR = "ADWIN"
CLASSIFIER = "random"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='Number of simulations to run.', default=N, type=int)
    parser.add_argument('-t', help='Number of instances to generate', default=T, type=int)
    parser.add_argument('--seed', help="Specify the seed for the Model's pseudorandom number generator", default=None,
                        type=int)
    parser.add_argument('-c', '--config', help="The name of the config file used to create the datastream.",
                        default=None, type=str)
    parser.add_argument('--detector', help="The detector to use..",
                        default=DETECTOR, type=str)
    parser.add_argument('--classifier', help="The classifier to use with detector.",
                        default=CLASSIFIER, type=str)

    parser.add_argument('-d', '--debug',
                        help='Sets the model to debug mode. Output printed to terminal will be verbose',
                        action='store_true')
    parser.add_argument('-l', '--log',
                        help='Tell the application to generate a log file for this run of the simulation.',
                        action='store_true')
    parser.add_argument('-r', '--record',
                        help='Tell the application to record all of the model data to a vegetation and agent csv file',
                        action='store_true')

    return parser.parse_args()


def run_experiment(args):
    predictor = classifers.RandomClassifier(seed=args.seed) if args.classifier == "random" \
        else classifers.ThresholdClassifier()

    bits = False
    if args.detector == "ADWIN":
        detector = drift.ADWIN
    elif args.detector == "KSWIN":
        detector = drift.KSWIN
    elif args.detector == "PAGE":
        detector = drift.PageHinkley
    elif args.detector == "EDDM":
        detector = drift.binary.EDDM
        bits = True
    else:
        detector = drift.binary.DDM
        bits = True

    model = args.config(predictor, args.seed)
    model.execute(args.t)

    stream = [x for x in model.get_stream()]

    detected = [False , False]
    for c in range(2):
        detector_instance = detector()
        for i in range(args.t):
            x, y, y_hat, t = stream[i]

            if y != c:  # Skip if label doesn't match class
                continue

            if bits:
                x = 0 if y == y_hat else 1  # 1 is error signal
            else:
                x = x[0]

            detector_instance.update(x)

            if detector_instance.drift_detected:
                detected[c] = True
                break

    return detected


def main():

    args = parse_args()

    if args.debug:
        np.seterr(invalid='raise')

    if args.config is not None:
        print(f"Data Stream Config Found, loading {args.config}.")
        args.config = runner.load_model_configuration(args.config)
    else:
        print(f"No Data Stream Config Found, using default stream instead.")

    none = 0
    one = 0
    both = 0

    for i in range(args.n):
        print(f"Running Experiment: {i}...")
        res = run_experiment(args)

        if res[0] and res[1]:
            both += 1
        elif res[0] or res[1]:
            one += 1
        else:
            none += 1

        print("Complete!")

    print(f"Results (n={args.n}): Both: {both / args.n}, One: {one / args.n}, None: {none / args.n}")
    print("Done!")


if __name__ == '__main__':
    main()
