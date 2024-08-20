import argparse

import pandas as pd
import sklearn.cluster as skcluster
import sklearn.metrics as skmetrics


KMIN = 2  # Repetitions
KMAX = 10  # Number of Instances
LABEL = "label"
INSTANCE = 'id'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Dataset to Import.', type=str)
    parser.add_argument('--kmin', help='Min K value', default=KMIN, type=int)
    parser.add_argument('--kmax', help='Max K value', default=KMAX, type=int)
    parser.add_argument('--label', help='Name of label in dataset', default=LABEL, type=str)
    parser.add_argument('--instance', help='Name of Instance ID in dataset', default=INSTANCE, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.d)
    print(df.head())

    del df[args.instance]
    del df[args.label]

    # Convert to Matrix
    instances = df.values
    print(instances.shape)

    for k in range(args.kmin, args.kmax + 1):
        model = skcluster.KMeans(n_clusters=k).fit(instances)
        labels = model.labels_

        print(f"K={k}: Silo: {skmetrics.silhouette_score(instances, labels, metric='euclidean')}")


if __name__ == '__main__':
    main()
