# Zero Noise Extrapolation with FiQCI EMS

## What is Zero Noise Extrapolation?

Zero Noise Extrapolation (ZNE) is an error mitigation technique that estimates the ideal (zero-noise) expectation value of a quantum circuit by intentionally amplifying the noise at several known levels and then extrapolating back to the zero-noise limit.

The core idea is:

1. **Run the original circuit** to obtain an expectation value at the base noise level.
2. **Amplify the noise** by creating scaled versions of the circuit (e.g., at scale factors 1, 3, 5).
3. **Extrapolate** the measured expectation values to the zero-noise point using a fitted model.

Scale factors may be **any real number ≥ 1** (e.g. `[1, 1.5, 2, 3]`), not just odd integers. See [Arbitrary scale factors](#arbitrary-scale-factors) below.

Because noise grows predictably with circuit depth, measuring at multiple noise levels reveals the trend, and extrapolation removes the noise contribution.

## Circuit Folding Methods

FiQCI EMS supports two methods for amplifying noise by increasing the effective circuit depth.

- **Local folding**: Local folding replaces individual two-qubit gates $G$ with $G G G$ (for scale factor 3), $G G G G G$ (for scale factor 5), and so on. Each gate is repeated `scale_factor` times in place.
    - Only two-qubit gates are folded (single-qubit gate errors are typically negligible).
    - The `fold_gates` parameter can restrict folding to specific gate names. If `None`, all two-qubit gates are folded, except MOVE (see [Devices with computational resonators](#devices-with-computational-resonators)).

- **Global folding**: Global folding appends the entire circuit and its inverse in alternating sequence. For a circuit $C$ with scale factor 3, the result is $C C^\dagger C$, and for scale factor 5: $C C^\dagger C C^\dagger C$.
    - This uniformly amplifies noise across all gates.
    - The `fold_gates` parameter is not applicable and will be ignored if set.

### Devices with computational resonators

On Deneb-class devices, `transpile_to_IQM` routes two-qubit interactions through a computational
resonator, inserting MOVE gates that transfer a qubit's state into the resonator and back. MOVE is
handled specially by folding.

Qiskit cannot invert a MOVE gate: it is deliberately defined without a decomposition so the
transpiler cannot break it apart, which also makes `inverse()` raise. FiQCI EMS substitutes an
internal self-inverse subclass during folding. This is exact rather than an approximation — MOVE is
an involution on its invariant subspace, so `MOVE MOVE` is the identity and `MOVE MOVE MOVE` is
equivalent to a single MOVE. The substituted gate keeps the MOVE name and type, so the circuit still
passes IQM's MOVE-sandwich validation.

**Local folding does not fold MOVE gates by default.** Folding a MOVE adds extra state transfers but
does not scale how long the state sits in the resonator, which is where the dominant error is, so the
noise would not be amplified by the requested scale factor. Only the CZ gates are folded, and MOVE is
excluded from the foldable-gate count used to compute the achieved scale factors. Pass
`fold_gates=["move"]` to fold MOVE anyway.

**Global folding always folds MOVE**, since it inverts the whole circuit and cannot hold individual
gates back. Prefer local folding on resonator devices unless you specifically want MOVE amplified.

### Arbitrary scale factors

Odd integers are the only scale factors reachable by *fully* folding every gate. Any other real value ≥ 1 (even integers, fractions) is **approximated** by partial folding:

- **Local**: every foldable gate is folded a base number of times, then a randomly-sampled subset is folded once more so the average folding matches the requested scale factor as closely as possible.
- **Global**: the circuit is fully folded for the integer part, then a suffix of the circuit is partially folded for the fractional remainder.

Because folding is discrete, small circuits may not reach the requested scale factor exactly. Extrapolation always uses the **achieved** scale factors as the x-axis (they depend only on each circuit's foldable-gate count, not the random seed, and equal the requested values exactly for odd integers). Your `scale_factors` configuration is never modified; instead, when any requested value can't be reached exactly the estimator warns at `run` and attaches both the requested and achieved values to the returned job:

```python
job = estimator.run(circuits, observables=observables)
job.requested_scale_factors()   # what you asked for, one list per circuit/observable pair
job.achieved_scale_factors()    # what folding realised (the extrapolation x-axis), same shape
```

Both accessors return a list of lists (one inner list per circuit/observable pair) and take an optional pair index. Pass a `seed` to make the random gate sampling reproducible.

#### When folding cannot separate the scale factors

Since folding is discrete, a circuit with few foldable gates can collapse several *distinct* requested scale factors onto the same achieved value. Extrapolation is then fitting fewer distinct x-values than it has points, which produces an unreliable fit — or, if every scale collapses to a single value, `nan`/`inf` expectation values. The most common cause is local folding on a circuit with **no two-qubit gates at all**, which cannot be folded locally, so every requested scale becomes 1.0.

The estimator cannot detect this before building the circuits, so it warns at `run` — after submission but before any results are fetched, so the run is still cancellable via the returned job handle:

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error")     # turn the warning into an exception if you prefer
    job = estimator.run(circuits, observables=observables)
```

Two distinct warnings are issued per circuit/observable pair:

- *every* requested scale collapsed onto one achieved value — no extrapolation is possible and the returned values will be meaningless. Cancel the job and either switch to `folding_method="global"` (which folds single-qubit gates too), widen the scale factors, or disable ZNE.
- *some* scales collapsed — the fit uses duplicated points and may be unreliable. Widen the scale factors if that is not intended.

Check `job.achieved_scale_factors()` to see exactly which values collapsed.

The returned job also carries `job.mitigator_options`, a snapshot of the full ZNE configuration (folding/extrapolation settings) in effect at submission, merged with the underlying backend mitigation settings. Unlike `estimator.mitigator_options`, this snapshot is frozen and reports what the run actually used even if you reconfigure the estimator afterwards.

### Per-circuit scale factors

`scale_factors` can be either a single flat list applied to every circuit, or a **list of lists**, one list per submitted circuit, so each circuit uses its own scale factors. The number of lists must match the number of circuit/observable pairs passed to `run`:

```python
estimator.zne(
    enabled=True,
    # circuit 0 uses [1, 3, 5]; circuit 1 uses [1, 2, 4]
    scale_factors=[[1, 3, 5], [1, 2, 4]],
)
job = estimator.run([circuit0, circuit1], observables=[obs0, obs1])
```

## Extrapolation Methods

After running circuits at each scale factor, the expectation values are extrapolated to zero noise. FiQCI EMS provides four extrapolation methods:

- **Exponential**: Fits an exponential decay model and works well when noise causes exponential decay of expectation values, which is common for depolarizing noise.

- **Richardson**: Uses Lagrange interpolation to compute exact coefficients that combine the measured values into a zero-noise estimate. This is a model-free method that makes no assumptions about the noise shape.

- **Polynomial**: Fits a polynomial of a given degree to the data. The degree defaults to `min(n_scales - 1, 2)` and can be set with the `extrapolation_degree` parameter.

- **Linear**: A special case of polynomial extrapolation with degree 1. Fits a straight line through the data points.

### Custom extrapolation functions

Instead of one of the built-in strings, `extrapolation_method` accepts a **user-defined callable**. It is invoked once per circuit/observable pair as `fn(expectation_values, scale_factors)` and must return a list of floats (the zero-noise estimate per observable):

- `expectation_values`: a list with one entry per scale factor; each entry is itself the list of per-observable expectation values measured at that scale (shape `(n_scales, n_obs)`, the same layout the built-in functions receive).
- `scale_factors`: the list of **achieved** scale factors for that pair (the extrapolation x-axis).

```python
import numpy as np

def my_linear_fit(expectation_values, scale_factors):
    y = np.asarray(expectation_values, dtype=float)  # (n_scales, n_obs)
    x = np.asarray(scale_factors, dtype=float)
    # Fit a line per observable and evaluate at x = 0.
    return [float(np.polyval(np.polyfit(x, y[:, j], 1), 0.0)) for j in range(y.shape[1])]

estimator.zne(enabled=True, scale_factors=[1, 3, 5], extrapolation_method=my_linear_fit)
```

`extrapolation_degree` is ignored when a callable is supplied.

#### Reporting standard errors from a custom extrapolation

A callable can also report the uncertainty of its zero-noise estimate, exactly as the built-in extrapolators do. If it accepts a `sigmas` keyword argument, it is additionally called with `sigmas=<per-scale shot standard errors>` (same `(n_scales, n_obs)` shape as `expectation_values`), and it may then return a `(values, standard_errors)` pair instead of just the values:

```python
def my_linear_fit(expectation_values, scale_factors, sigmas=None):
    y = np.asarray(expectation_values, dtype=float)  # (n_scales, n_obs)
    x = np.asarray(scale_factors, dtype=float)
    values, errors = [], []
    for j in range(y.shape[1]):
        coeffs = np.polyfit(x, y[:, j], 1)
        values.append(float(np.polyval(coeffs, 0.0)))
        # Row of the least-squares operator giving the intercept, so Var(E(0)) = sum_i a_i^2 sigma_i^2.
        a = np.linalg.pinv(np.vander(x, 2))[-1, :]
        errors.append(float(np.sqrt(np.sum(a**2 * np.asarray(sigmas)[:, j] ** 2))))
    return values, errors

estimator.zne(enabled=True, scale_factors=[1, 3, 5], extrapolation_method=my_linear_fit)
job.standard_errors(0)["zne_extrapolation_error"]  # the errors the callable returned
```

The returned `standard_errors` must have one entry per expectation value; a length mismatch raises `ValueError`. The values are surfaced unchanged as the `zne_extrapolation_error` and `total` keys of {meth}`~fiqci.ems.primitives.fiqci_estimator.FiQCIEstimatorJob.standard_errors`. Callables that take no `sigmas` argument, or return only values, leave both keys `None` — `shot_error` is still measured and reported. A callable may also return `(values, None)` to report values without standard errors, which is treated the same as returning the values alone. Anything else — a return value that cannot be read as a sequence of floats — raises `TypeError` naming what was returned. Because the built-in extrapolators follow this same convention, they can be passed as callables directly (e.g. `extrapolation_method=richardson_extrapolation`) and their propagated errors come through as usual.

## Usage

### Via Mitigation Level

Setting `mitigation_level=3` enables ZNE with default settings (local folding, scale factors [1, 3, 5], exponential extrapolation):

```python
from fiqci.ems import FiQCIEstimator

estimator = FiQCIEstimator(backend=backend, mitigation_level=3)
```

### Manual Configuration

For fine-grained control, enable ZNE explicitly via the {meth}`~fiqci.ems.FiQCIEstimator.zne` method:

```python
estimator = FiQCIEstimator(backend=backend, mitigation_level=1)

estimator.zne(
    enabled=True,
    scale_factors=[1, 3, 5],
    folding_method="global",
    extrapolation_method="richardson",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | - | Enable or disable ZNE. |
| `fold_gates` | `list[str] \| None` | `None` | Gate names to fold (local folding only). `None` folds all two-qubit gates except MOVE. |
| `scale_factors` | `list[float] \| list[list[float]]` | `[1, 3, 5]` | Real numbers ≥ 1 specifying the noise scale levels (odd integers fold exactly; other values are approximated). At least two are required. May be a list of lists to give each submitted circuit its own scale factors. |
| `folding_method` | `str` | `"local"` | `"local"` or `"global"`. |
| `extrapolation_method` | `str \| Callable` | `"exponential"` | `"exponential"`, `"richardson"`, `"polynomial"`, `"linear"`, or a custom callable `fn(expectation_values, scale_factors[, sigmas]) -> list[float] \| tuple[list[float], list[float]]` (see [Custom extrapolation functions](#custom-extrapolation-functions)). |
| `extrapolation_degree` | `int \| None` | `None` | Polynomial degree (only for `"polynomial"` extrapolation). |
| `seed` | `int \| None` | `None` | Seed for the random gate sampling used to approximate non-odd-integer scale factors. |

## Extrapolation Uncertainty

Each extrapolator here is a fixed *linear* map from the per-scale measurements to the zero-noise estimate, $E(0) = \sum_i a_i\, E(\lambda_i)$, where the coefficients $a_i$ depend only on the scale factors and the method. The shot noise of each measured point therefore propagates analytically to the extrapolated value:

$$\mathrm{Var}\big(E(0)\big) = \sum_i a_i^2\, \sigma_i^2 .$$

This standard error is available through {meth}`~fiqci.ems.primitives.fiqci_estimator.FiQCIEstimatorJob.standard_errors` as the `zne_extrapolation_error` key:

```python
estimator.zne(enabled=True, extrapolation_method="linear")
job = estimator.run(qc, observables=obs, shots=2048)

job.standard_errors(0)["zne_extrapolation_error"]  # propagated SE of the zero-noise estimate
```

Because extrapolation evaluates the fit *outside* the measured range (at $\lambda = 0$), the coefficients $a_i$ are large and alternate in sign, so the extrapolation error is typically **larger** than the raw shot error, the price ZNE pays in variance for reducing bias. Only this statistical (shot-propagated) error is reported. The systematic *model* error from choosing a particular extrapolation shape is not quantified.

A user-defined extrapolation callable can report its own standard errors through the same key; see [Reporting standard errors from a custom extrapolation](#reporting-standard-errors-from-a-custom-extrapolation).

See {doc}`FiQCIEstimator standard errors <FiQCIEstimatorUsage>` for the full set of error keys.

## Examples

- [Zero Noise Extrapolation Example](../notebooks/zero_noise_extrapolation_example). Runnable notebook demonstrating ZNE with default and custom settings.

## References

- Temme, K., Bravyi, S., Gambetta, J. M., "Error Mitigation for Short-Depth Quantum Circuits", [https://arxiv.org/abs/1612.02058](https://arxiv.org/abs/1612.02058)
- Li, Y., Benjamin, S. C., "Efficient Variational Quantum Simulator Incorporating Active Error Minimization", [https://arxiv.org/abs/1611.09301](https://arxiv.org/abs/1611.09301)
- Pegah Mohammadipour, Xiantao Li., "Direct Analysis of Zero-Noise Extrapolation: Polynomial Methods, Error Bounds, and Simultaneous Physical-Algorithmic Error Mitigation" [https://arxiv.org/abs/2502.20673](https://arxiv.org/abs/2502.20673)
