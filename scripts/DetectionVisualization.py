import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Name of json file containing input data' , type=str)
    parser.add_argument('-o', '--output', help="The name of the output folder to write figures to.", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input) as f:  # TODO: This should probably be checked more thoroughly
        records = json.load(f)

    results = {}

    # Get pvals
    for ex in records:
        for c in ex:
            if c not in results:
                results[c] = []

            results[c].append(ex[c]['p'])

    fig, ax = plt.subplots()
    for c in results:
        results[c] = np.array(results[c])
        print(results[c].shape)
        med = np.mean(results[c], axis=0)
        ax.plot(np.arange(len(med)), med, label=c)
        std = np.std(results[c], axis=0)
        ax.fill_between(np.arange(len(std)),  med - std, med + std, alpha=0.2)
    ax.hlines(y=0.05, xmin=0.0, xmax=100, linewidth=2, color='r')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
