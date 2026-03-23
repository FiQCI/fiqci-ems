import numpy as np

from typing import Iterable

def exponential_extrapolation(expectation_values, scale_factors):
    """
    Perform exponential extrapolation to estimate the zero-noise value.

    Args:
        expectation_values: A list of expectation values corresponding to different noise levels.

    Returns:
        The extrapolated zero-noise expectation value.
    """
    if len(expectation_values) < 2:
        raise ValueError("At least two expectation values are required for exponential extrapolation.")


    x = np.array(scale_factors)  # Noise scale factors
    y = np.array(expectation_values)

    negative_index = []
    for i, val in enumerate(y[0]) if isinstance(y[0], Iterable) else enumerate(y):
        if val < 0:
            negative_index.append(i)

    y = np.abs(y)

    # Fit an exponential curve to the data points
    coeffs = np.polyfit(x, np.log(y), 1)
    a, b = coeffs

    # Extrapolate to zero noise (x=0)
    zero_noise_value = np.exp(b)

    zero_noise_value = [-v if i in negative_index else v for i, v in enumerate(zero_noise_value)]

    return [float(v) for v in zero_noise_value]

def richardson_extrapolation(expectation_values, scales, degree=None):
    """
    Polynomial (Richardson) extrapolation to estimate the zero-noise value.

    Args:
        expectation_values: Array-like of shape (n_scales, n_obs) or (n_scales,)
        scales: Noise scale factors used (e.g., [1, 3, 5])
        degree: Optional polynomial degree; defaults to min(n_scales-1, 2)

    Returns:
        Zero-noise estimate(s) per observable.
    """

    y = np.asarray(expectation_values, dtype=float)
    x = np.asarray(scales, dtype=float)

    if y.ndim == 1:
        y = y[:, None]

    if len(x) != y.shape[0]:
        raise ValueError("Length mismatch between scales and expectation_values.")

    deg = degree if degree is not None else min(y.shape[0] - 1, 2)
    out = np.empty(y.shape[1])

    for j in range(y.shape[1]):
        mask = np.isfinite(y[:, j])
        if mask.sum() < 2:
            out[j] = np.nan
            continue
        coeffs = np.polyfit(x[mask], y[mask, j], deg)
        out[j] = np.polyval(coeffs, 0.0)

    res = out if out.size > 1 else out[0]

    return [float(v) for v in res]