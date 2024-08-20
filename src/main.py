import argparse
import json
import numpy as np

import ExperimentRunner as runner

N = 1  # Repetitions
T = 100000  # Number of Instances
WARMUP = 5  # Number of Trials before statistical test is used
SIGMA = 0.001  # Influence Strength
F = 0.5
OFFSET = 0.0  # Offset of F
TAU = 1000  # Length of Trial
P = 0.05  # Confidence Interval
W1 = 100  # Window from beginning and end of trial to take instances from when constructing d1 and d2 distributions
MIX = 0.0

IMAG = 1.0
IFREQ = 1
IRESET = -1
ITYPE = "none"

OUTPUT = "./results.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', help='Number of simulations to run.', default=N, type=int)
    parser.add_argument('-t', help='Number of instances to generate', default=T, type=int)
    parser.add_argument('--warmup', help='Number of Trials that need to be run before statistical testing will start',
                        default=WARMUP, type=int)
    parser.add_argument('--sigma', help='Strength of the Performative Drift', default=SIGMA, type=float)
    parser.add_argument('--tau', help='The length of a trial period. Value MUST integer divide with T',
                        default=TAU, type=int)
    parser.add_argument('-f', help='Length of a Checkerboard in the feature space.', default=F, type=float)
    parser.add_argument('--offset', help='Offset of Checkerboard in the feature space.', default=OFFSET, type=float)
    parser.add_argument('-p', help='Confidence Interval to Detect Performative Drift.', default=P, type=float)
    parser.add_argument('--wone', help='Window from beginning and end of trial to take instances '
                                       'from when constructing d1 and d2 distributions', default=W1, type=int)
    parser.add_argument('--seed', help="Specify the seed for the Model's pseudorandom number generator", default=None,
                        type=int)
    parser.add_argument('-c', '--config', help="The name of the config file used to create the datastream.",
                        default=None, type=str)
    parser.add_argument('-o', '--output', help="The name of the output file to write results to.",
                        default=OUTPUT, type=str)
    parser.add_argument('--classifier', help="The classifier to use with detector.",
                        default=None, type=str)
    parser.add_argument('--mix', help="The portion of instances to divide between classifier and detector.",
                        default=MIX, type=float)

    parser.add_argument('--imag', help="Magnitude of Intrinsic Drift.",
                        default=IMAG, type=float)
    parser.add_argument('--ifreq', help="Frequency of Intrinsic Drift.",
                        default=IFREQ, type=int)
    parser.add_argument('--ireset', help="Frequency at which drift should be reset.",
                        default=IRESET, type=int)
    parser.add_argument('--itype', help="Type of Intrinsic Drift (gradual or sudden).",
                        default=ITYPE, type=str)

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


def run_experiment(i: int, args):
    return runner.run_experiment(args.f, args.t, args.tau, args.wone, args.config, offset=args.offset,
                                 classifier=args.classifier, mix=args.mix, itype=args.itype,
                                 imag=args.imag, ifreq=args.ifreq, ireset=args.ireset)


def write_to_file(filename, data):
    # Serializing json
    json_object = json.dumps(data, indent=4)

    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def main():

    args = parse_args()
    # TODO: Validate argruments.

    if args.debug:
        np.seterr(invalid='raise')

    if args.config is not None:
        print(f"Data Stream Config Found, loading {args.config}.")
        args.config = runner.load_model_configuration(args.config)
    else:
        print(f"No Data Stream Config Found, using default stream instead.")

    data = []
    for i in range(args.n):
        print(f"Running Experiment: {i}...")
        data.append(run_experiment(i, args))
        print("Complete!")

    print(f"Writing Results to {args.output}...")
    write_to_file(args.output, data)
    print("Done!")


if __name__ == '__main__':
    main()
