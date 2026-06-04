"""
Extrapolation methods for Zero-Noise Extrapolation.
"""

import numpy as np


def exponential_extrapolation(
	expectation_values: list[list[float]], scale_factors: list[int], eps: float = 1e-9
) -> list[float]:
	"""
	Perform exponential extrapolation to estimate the zero-noise value.

	Fits y = sign * exp(b) * exp(a * x) in log-space per observable. Magnitudes are
	floored relative to each column's largest value before taking the log, so values
	that are ~0 (or whose sign flips due to noise) can't produce log(0) = -inf or
	dominate the linear fit.

	Args:
	    expectation_values: Expectation values of shape (n_scales, n_obs) or (n_scales,).
	    scale_factors: Noise scale factors corresponding to different noise levels.
	    eps: Magnitude floor as a fraction of each column's maximum magnitude.

	Returns:
	    The extrapolated zero-noise expectation value(s).
	"""
	if len(expectation_values) < 2:
		raise ValueError("At least two expectation values are required for exponential extrapolation.")

	x = np.asarray(scale_factors, dtype=float)
	y = np.asarray(expectation_values, dtype=float)

	if y.ndim == 1:
		y = y[:, None]

	# The sign comes from the lowest-noise (smallest scale) measurement, the most reliable point.
	ref = y[np.argmin(x), :]

	out = np.empty(y.shape[1])
	for j in range(y.shape[1]):
		mag = np.abs(y[:, j])
		scale = mag.max()
		if scale <= eps:
			# Signal is indistinguishable from zero; the exponential model is meaningless,
			# so report zero rather than fitting noise.
			out[j] = 0.0
			continue
		# Floor magnitudes relative to the column scale to keep log() finite and stop
		# near-zero points from dominating the linear fit.
		mag = np.maximum(mag, eps * scale)
		b = np.polyfit(x, np.log(mag), 1)[1]
		sign = np.sign(ref[j]) or 1.0
		out[j] = sign * np.exp(b)

	return [float(v) for v in out]


def richardson_extrapolation(expectation_values: list[list[float]], scales: list[int]) -> list[float]:
	"""
	Richardson extrapolation to estimate the zero-noise value.

	Computes exact Lagrange interpolation coefficients evaluated at x=0:
	cᵢ = ∏_{j≠i} λⱼ / (λⱼ - λᵢ) and returns E(0) = Σᵢ cᵢ · E(λᵢ).

	Args:
	    expectation_values: Array-like of shape (n_scales, n_obs) or (n_scales,)
	    scales: Noise scale factors used (e.g., [1, 3, 5])

	Returns:
	    Zero-noise estimate(s) per observable.
	"""

	y = np.asarray(expectation_values, dtype=float)
	x = np.asarray(scales, dtype=float)

	if y.ndim == 1:
		y = y[:, None]

	if len(x) != y.shape[0]:
		raise ValueError("Length mismatch between scales and expectation_values.")

	n = len(x)
	coeffs = np.empty(n)
	for i in range(n):
		mask = np.arange(n) != i
		num = np.prod(x[mask])
		den = np.prod(x[mask] - x[i])
		coeffs[i] = num / den

	out = coeffs @ y
	return [float(v) for v in out]


def polynomial_extrapolation(
	expectation_values: list[list[float]], scales: list[int], degree: int | None = None
) -> list[float]:
	"""
	Polynomial least-squares extrapolation to estimate the zero-noise value.

	Fits a polynomial of the given degree to the (scale, expectation_value)
	data and evaluates it at x=0.

	Args:
	    expectation_values: Array-like of shape (n_scales, n_obs) or (n_scales,)
	    scales: Noise scale factors used (e.g., [1, 3, 5])
	    degree: Polynomial degree. Defaults to min(n_scales - 1, 2).

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

	return [float(v) for v in out]
