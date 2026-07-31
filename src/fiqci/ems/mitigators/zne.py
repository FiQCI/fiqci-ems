"""
Extrapolation methods for Zero-Noise Extrapolation.
"""

import numpy as np


def _ls_intercept_row(x: np.ndarray, degree: int) -> np.ndarray:
	"""Row vector ``a`` such that the least-squares polynomial fit evaluated at x=0 equals ``a @ y``.

	The fit is linear in the data, so the zero-noise estimate is a fixed linear combination of the
	per-scale values whose coefficients depend only on ``x`` and the polynomial degree. ``a`` is the
	constant-term row of the prediction operator ``(VᵀV)⁻¹ Vᵀ`` for the Vandermonde matrix ``V``.
	"""
	# np.polyfit orders coefficients highest power first, so the constant term is the last row.
	vander = np.vander(x, degree + 1)
	pinv = np.linalg.pinv(vander)
	return pinv[-1, :]


def exponential_extrapolation(
	expectation_values: list[list[float]],
	scale_factors: list[float],
	eps: float = 1e-9,
	sigmas: list[list[float]] | None = None,
) -> list[float] | tuple[list[float], list[float]]:
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
	    sigmas: Optional per-scale shot standard errors, same shape as ``expectation_values``.
	        When provided, the per-observable standard error of the extrapolated value is returned
	        alongside the values.

	Returns:
	    The extrapolated zero-noise expectation value(s), or ``(values, standard_errors)`` when
	    ``sigmas`` is provided.
	"""
	if len(expectation_values) < 2:
		raise ValueError("At least two expectation values are required for exponential extrapolation.")

	x = np.asarray(scale_factors, dtype=float)
	y = np.asarray(expectation_values, dtype=float)

	if y.ndim == 1:
		y = y[:, None]

	# The sign comes from the lowest-noise (smallest scale) measurement, the most reliable point.
	ref = y[np.argmin(x), :]

	# Row of the degree-1 least-squares operator that yields the log-space intercept b = a @ log(mag).
	a = _ls_intercept_row(x, 1) if sigmas is not None else None
	sig = np.asarray(sigmas, dtype=float) if sigmas is not None else None
	if sig is not None and sig.ndim == 1:
		sig = sig[:, None]

	out = np.empty(y.shape[1])
	errs = np.empty(y.shape[1])
	for j in range(y.shape[1]):
		mag = np.abs(y[:, j])
		scale = mag.max()
		if scale <= eps:
			# Signal is indistinguishable from zero; the exponential model is meaningless,
			# so report zero rather than fitting noise.
			out[j] = 0.0
			errs[j] = 0.0
			continue
		# Floor magnitudes relative to the column scale to keep log() finite and stop
		# near-zero points from dominating the linear fit.
		mag = np.maximum(mag, eps * scale)
		b = np.polyfit(x, np.log(mag), 1)[1]
		sign = np.sign(ref[j]) or 1.0
		out[j] = sign * np.exp(b)
		if a is not None and sig is not None:
			# Propagate shot errors into log-space (sigma_log = sigma / |y|), through the linear
			# intercept (Var(b) = sum a_i^2 sigma_log_i^2), then by the delta method onto
			# E0 = sign * exp(b): SE = |E0| * sqrt(Var(b)).
			sigma_log = sig[:, j] / mag
			var_b = float(np.sum((a**2) * (sigma_log**2)))
			errs[j] = abs(out[j]) * np.sqrt(var_b)

	values = [float(v) for v in out]
	if sigmas is None:
		return values
	return values, [float(e) for e in errs]


def richardson_extrapolation(
	expectation_values: list[list[float]], scales: list[float], sigmas: list[list[float]] | None = None
) -> list[float] | tuple[list[float], list[float]]:
	"""
	Richardson extrapolation to estimate the zero-noise value.

	Computes exact Lagrange interpolation coefficients evaluated at x=0:
	cᵢ = ∏_{j≠i} λⱼ / (λⱼ - λᵢ) and returns E(0) = Σᵢ cᵢ · E(λᵢ).

	Args:
	    expectation_values: Array-like of shape (n_scales, n_obs) or (n_scales,)
	    scales: Noise scale factors used (e.g., [1, 3, 5])
	    sigmas: Optional per-scale shot standard errors, same shape as ``expectation_values``.
	        When provided, the per-observable standard error of the extrapolated value is returned
	        alongside the values.

	Returns:
	    Zero-noise estimate(s) per observable, or ``(values, standard_errors)`` when ``sigmas`` is
	    provided.
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
	values = [float(v) for v in out]
	if sigmas is None:
		return values

	# E(0) = sum_i c_i y_i is linear in the data, so Var(E(0)) = sum_i c_i^2 sigma_i^2.
	sig = np.asarray(sigmas, dtype=float)
	if sig.ndim == 1:
		sig = sig[:, None]
	var = (coeffs**2) @ (sig**2)
	return values, [float(np.sqrt(v)) for v in var]


def polynomial_extrapolation(
	expectation_values: list[list[float]],
	scales: list[float],
	degree: int | None = None,
	sigmas: list[list[float]] | None = None,
) -> list[float] | tuple[list[float], list[float]]:
	"""
	Polynomial least-squares extrapolation to estimate the zero-noise value.

	Fits a polynomial of the given degree to the (scale, expectation_value)
	data and evaluates it at x=0.

	Args:
	    expectation_values: Array-like of shape (n_scales, n_obs) or (n_scales,)
	    scales: Noise scale factors used (e.g., [1, 3, 5])
	    degree: Polynomial degree. Defaults to min(n_scales - 1, 2).
	    sigmas: Optional per-scale shot standard errors, same shape as ``expectation_values``.
	        When provided, the per-observable standard error of the extrapolated value is returned
	        alongside the values.

	Returns:
	    Zero-noise estimate(s) per observable, or ``(values, standard_errors)`` when ``sigmas`` is
	    provided.
	"""

	y = np.asarray(expectation_values, dtype=float)
	x = np.asarray(scales, dtype=float)

	if y.ndim == 1:
		y = y[:, None]

	if len(x) != y.shape[0]:
		raise ValueError("Length mismatch between scales and expectation_values.")

	deg = degree if degree is not None else min(y.shape[0] - 1, 2)
	out = np.empty(y.shape[1])

	sig = np.asarray(sigmas, dtype=float) if sigmas is not None else None
	if sig is not None and sig.ndim == 1:
		sig = sig[:, None]
	errs = np.empty(y.shape[1])

	for j in range(y.shape[1]):
		mask = np.isfinite(y[:, j])
		if mask.sum() < 2:
			out[j] = np.nan
			errs[j] = np.nan
			continue
		coeffs = np.polyfit(x[mask], y[mask, j], deg)
		out[j] = np.polyval(coeffs, 0.0)
		if sig is not None:
			# The fit evaluated at x=0 is linear in the data: E(0) = a @ y, with a the constant-term
			# row of the least-squares operator. Var(E(0)) = sum_i a_i^2 sigma_i^2.
			a = _ls_intercept_row(x[mask], deg)
			errs[j] = float(np.sqrt(np.sum((a**2) * (sig[mask, j] ** 2))))

	values = [float(v) for v in out]
	if sigmas is None:
		return values
	return values, [float(e) for e in errs]
