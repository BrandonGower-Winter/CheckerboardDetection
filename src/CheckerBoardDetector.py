import numpy as np
import pandas as pd
import random
import river.base.classifier as base
import river.base.typing as typing


class CheckerBoard1DUniform (base.MiniBatchClassifier):

    _labels = [1, 0, 0, 1]

    def __init__(self, feature_split: float, time_split: int, flip: float = 0.0, feature_name=None, moment=None,
                 offset: float = 0.0, seed: int = None):
        super(CheckerBoard1DUniform, self).__init__()

        self._feature_split = feature_split
        self._offset = offset
        self._time_split = time_split

        self._feature_name = feature_name
        self._moment = 't' if moment is None else moment

        self._flip = flip
        self._random = random.Random(seed)

    def learn_one(self, x: dict, y: typing.ClfTarget) -> base.Classifier:
        return self

    def predict_proba_one(self, x: dict) -> dict[typing.ClfTarget, float]:
        raise NotImplementedError

    def predict_one(self, x: dict, **kwargs) -> typing.ClfTarget | None:
        """Predict the label of a set of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The predicted label.

        """
        if self._feature_name is None:
            if 0 not in x:
                self._feature_name = [key for key in x.keys()][0]  # TODO: make sure dict is valid
            else:
                self._feature_name = 0

        f_id = np.floor((x[self._feature_name] + self._offset) / self._feature_split) % 2
        t_id = np.floor(x[self._moment] / self._time_split) % 2  # TODO: Is this the best way to get time?
        y_hat = CheckerBoard1DUniform._labels[int(f_id) + int(2*t_id)]

        if self._random.random() < self._flip:
            y_hat = 1 - y_hat

        return y_hat

    def __call__(self, x, y, t):
        instance = {i: x[i] for i in range(len(x))}
        instance['t'] = t
        return self.predict_one(instance)

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        """Update the model with a mini-batch of features `X` and boolean targets `y`.

        Parameters
        ----------
        X
            A dataframe of features.
        y
            A series of boolean target values.

        """
        pass

    def predict_proba_many(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        """Predict the outcome for each given sample.

        Parameters
        ----------
        X
            A dataframe of features.

        Returns
        -------
        The predicted labels.

        """
        return X.apply(self.predict_one, axis=1)

    @property
    def _supervised(self):
        return False
