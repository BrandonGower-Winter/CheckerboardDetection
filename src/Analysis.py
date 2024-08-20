import numpy as np

from scipy.integrate import quad
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.neighbors import KernelDensity


def make_kde(x: np.ndarray, kernel: str = "gaussian", bandwidth: float = 0.75) -> KernelDensity:
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    if len(x) > 0:
        kde.fit(x)
    else:
        kde.fit([[0]])

    return kde


def make_distribution(x: np.ndarray, kde: KernelDensity = None, kernel: str = "gaussian",
                      bandwidth: float = 0.1) -> np.ndarray:

    if len(x) > 0:
        return np.exp(
            make_kde(x, kernel=kernel, bandwidth=bandwidth).score_samples(x)
            if kde is None else kde.score_samples(x)
        )
    else:
        return np.ndarray([0])


def make_rescaled_distribution(x: np.ndarray, n_start: int, n_end: int, kde: KernelDensity = None,
                               kernel: str = "gaussian", bandwidth: float = 0.1) -> (np.ndarray, KernelDensity, float):
    kde = kde if kde is not None else make_kde(x, kernel, bandwidth)

    # Calculate the Mean (From: https://stackoverflow.com/questions/55788868/
    # how-to-return-mean-value-or-expectation-value-of-a-distribution-estimated-via
    pdf = lambda x: np.exp(kde.score_samples([[x]]))[0]
    mean = quad(lambda x: x * pdf(x), a=-np.inf, b=np.inf)[0]
    # Check to see if log densities need to be used
    return (make_distribution(x, kde=kde) * n_start - mean) / (n_start - 1), kde, mean


def shape_data_to_rescaled_kde(x: np.ndarray, kde: KernelDensity, mean: float, n_start: int, n_end: int) -> float:
    result = kde.score_samples(x[:, np.newaxis]) if len(x) > 0 else np.array([[0]])
    return np.exp((result * n_start - mean) / (n_start - 1))


def calculate_change_distribution(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    return d2 - d1


def test_distributions(d1: np.ndarray, d2: np.ndarray, test: str = 'mannwhitney') -> (float, float):
    if test == 'wilcoxon':
        return wilcoxon(d1, d2)
    return mannwhitneyu(d1, d2)


