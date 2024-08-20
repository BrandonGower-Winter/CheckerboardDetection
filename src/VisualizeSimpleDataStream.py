import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from Analysis import test_distributions

THRESHOLD = 0.01
START = 0
END = 100
WINDOW_SIZE = 100
TEST_MIN = 100

NUM_TESTS = 1


def test_for_significance(d1, d2) -> int:
    a = []
    b = []

    return 1 if test_distributions(d1[START:END], d2[START:END]).pvalue < THRESHOLD else -1

    w_size = len(d1) - NUM_TESTS + 1
    if w_size < 0:
        return -1

    for i in range(START, END):

        if len(d1) <= i or len(d2) <= i:
            break

        a.append(d1[i])
        b.append(d2[i])

        while len(a) > w_size:
            a.pop(0)

        while len(b) > w_size:
            b.pop(0)

        if len(a) >= w_size:
            if test_distributions(a, b).pvalue < THRESHOLD:
                return START + i

    return -1


def main():
    # Look through all files
    for root, dirs, files in os.walk(sys.argv[1]):
        for f in files:

            if f.startswith("."):
                continue

            # Create Counters for Each Category of Detection
            both = 0
            none = 0
            one = 0
            total = 0
            # Load File
            with open(os.path.join(root, f)) as d:
                d = json.load(d)
                # Extract Each Simulation
                for run in d:
                    t_one = test_for_significance(run['1']['a'], run['1']['b'])
                    t_zero = test_for_significance(run['0']['a'], run['0']['b'])

                    if t_one > -1 and t_zero > -1:
                        both += 1
                    elif (t_one > -1 and t_zero == -1) or (t_one == -1 and t_zero > -1):
                        one += 1
                    else:
                        none += 1

                    total += 1

            print(f[0:-5])
            print(f'None: {none / total}')
            print(f'One: {one / total}')
            print(f'Both: {both / total}')
            print()


if __name__ == '__main__':
    main()
