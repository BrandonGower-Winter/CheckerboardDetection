# CheckerBoard Performative Drift Detection (CB-PDD)

CB-PDD is a performative drift detector. 
The implementation details of which can be found in the paper
titled "Identifying Predictions that Influence the Future: Detecting Performative Concept
Drift in Data Streams"

## Installation

To install CB-PDD, open a terminal, navigate to the main
directory "CheckerboardDetection" and type:

```bask
> make
```

This will build a virtual environment that will allow
you to run all the code present in the repository.

To activate the virtual environment, type:

```bash
> source ./venv/bin/activate
```

## Usage

To run CB-PDD, type the following in the terminal:

```bash
> python src/main.py -n N -t T -f F --tau TAU 
--config PATH/TO/CONFIG.json -o OUTPUT_FILE.json --debug 
```

where `N` is the number of experiments to run, `T` is the
number of instances to generate, `F` is the value
of the `f` parameter, `TAU` is the value of the `tau` parameter
and `--config` is the path to the config file that will be used
to build the intial data-streams.

```bash
> python src/main.py --help 
```

Can also be used to get a list of all available parameters
and their descriptions.

``--seed SEED`` can be used to specify a seed for pseudo-random
number generation. This should be used with `-n 1`.

### Intrinsic Drift

To simulate intrinsic drift, the following options are available:


For Sudden Drift:
```python
--itype sudden --ifreq FREQ
```

where `FREQ` is the number of instances that should be generated
before a drift event is triggered.

For Incremental / Gradual Drift:
```bash
 --itype gradual --ifreq 1 --imag MAG  --ireset RESET
```

where `MAGE` is the maximum velocity of the incremental drift
and `RESET` is the number of instances that should be generated
before the new random velocities are generated.

### Custom Configs

CB-PDD comes with several configuration already (See
the configs directory).

In general, the template for a config is:

```json
{
    "id": ID,
    "type": "multi_gauss",
    "label": LABEL,
    "lower": -1.0,
    "upper": 1.0,
    "n": C,
    "y_pos": 0.0,
    "sigma": SIGMA,
    "weights": "random",
    "weight_delta": STRENGTH,
    "move_delta": 0.0,
    "x_drift": 0.0,
    "feedback_type": LOOP_TYPE,
    "decay_feedback": true
}
```
where `ID` is the name for the group of centroids, `LABEL`
is the classification label that they produce, `C` is the 
number of centroids that are produced, `SIGMA` is the spread
of the Gaussian associated with each centroid, `STRENGTH`,
is the strength of the performative drift that will be induced,
and `LOOP_TYPE` is the type of feedback loop to induce. For a
self-fulfilling feedback loop, use `[1, 0, 0, 0]` and for a 
self-defeating feedback loop, use `[0, 1, 0, 0]`.

These centroid clusters can be combined with others
to create the intended initial distribution that need to be
evaluated.

## Viewing Results

To view the results of a set of experiments, use the following:

```bash
>  python ./src/VisualizeSimpleDataStream.py 
PATH_TO_DIRECTORY 
```

where `PATH_TO_DIRECTORY` is the directory that contains all
of the `.json` outputs produced by `main.py`. It will produce
a result that looks something like:

```text
FILE_NAME.json
None: VAL0
One: VAL1
Both: VAL2
```

where `VAL0`, `VAL1` and `VAL2` are the rates of no detection, 
only one class detected and both classes detected respectively.

## Running CB-PDD on datasets

To run CB-PDD on a dataset, use:

```bash
> src/main_dataset.py -n N -t T -f F --tau TAU 
-d /PATH/TO/DATASET.csv --instance ID 
--label LABEL -o OUTPUT.json --debug --sigma SIGMA
```

where `PATH/TO/DATASET.csv` is the dataset to use,
`ID` is the name of the column that identifies each instance,
`LABEL` is the name of the column that contains the label
for each instance and `SIGMA` is the desired performative
drift strength.

## Using AgentStream with Traditional Drift Detectors

To evaluate traditional drift detectors, type the following:

```bash
> python ./src/main_drift_detectors.py -n N -t T 
--config CONFIG.json --debug --classifier CLASSIFIER 
--detector DETECTOR
```

where `CLASSIFIER` is the type of classifier used with the detector
(either "random" or "threshold") and `DETECTOR` is the name
of the detector to use (either "DDM", "ADWIN", "KSWIN" or
"PAGE").