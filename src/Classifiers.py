import numpy as np
import pandas as pd
import random
import river.base.classifier as base
import river.base.typing as typing


class RandomClassifier(base.MiniBatchClassifier):

    def __init__(self, n_classes: int = 2, seed: int = None):
        super(RandomClassifier, self).__init__()

        self._n_classes = n_classes
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
        return self._random.randint(0, self._n_classes)

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


class ThresholdClassifier(base.MiniBatchClassifier):

    def __init__(self, threshold: float = 0.0):
        super(ThresholdClassifier, self).__init__()
        self._threshold = threshold

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
        return 0 if x[0] < self._threshold else 1

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


class Mixer(base.MiniBatchClassifier):

    def __init__(self, mix: float, model_one: base.MiniBatchClassifier, model_two: base.MiniBatchClassifier,
                 seed: int = None):
        super(Mixer, self).__init__()

        self._mix = mix
        self._model_one = model_one
        self._model_two = model_two
        self._random = random.Random(seed)

    def learn_one(self, x: dict, y: typing.ClfTarget) -> base.Classifier:
        # Maybe mix this too?
        self._model_one.learn_one(x, y)
        self._model_two.learn_one(x, y)
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
        if self._random.random() < self._mix:
            return self._model_one.predict_one(x, **kwargs)
        else:
            return self._model_two.predict_one(x, **kwargs)

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
        return self._model_one._supervised() or self._model_two._supervised()
