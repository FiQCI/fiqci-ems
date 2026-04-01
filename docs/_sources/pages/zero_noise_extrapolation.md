# Zero Noise Extrapolation with FiQCI EMS

## What is Zero Noise Extrapolation?

Zero Noise Extrapolation (ZNE) is an error mitigation technique that estimates the ideal (zero-noise) expectation value of a quantum circuit by intentionally amplifying the noise at several known levels and then extrapolating back to the zero-noise limit.

The core idea is:

1. **Run the original circuit** to obtain an expectation value at the base noise level.
2. **Amplify the noise** by creating scaled versions of the circuit (e.g., at scale factors 1, 3, 5).
3. **Extrapolate** the measured expectation values to the zero-noise point using a fitted model.

Because noise grows predictably with circuit depth, measuring at multiple noise levels reveals the trend, and extrapolation removes the noise contribution.

## Circuit Folding Methods

FiQCI EMS supports two methods for amplifying noise by increasing the effective circuit depth.

- **Local folding**: Local folding replaces individual two-qubit gates $G$ with $G G G$ (for scale factor 3), $G G G G G$ (for scale factor 5), and so on. Each gate is repeated `scale_factor` times in place.
    - Only two-qubit gates are folded (single-qubit gate errors are typically negligible).
    - The `fold_gates` parameter can restrict folding to specific gate names. If `None`, all two-qubit gates are folded.

- **Global folding**: Global folding appends the entire circuit and its inverse in alternating sequence. For a circuit $C$ with scale factor 3, the result is $C C^\dagger C$, and for scale factor 5: $C C^\dagger C C^\dagger C$.
    - This uniformly amplifies noise across all gates.
    - The `fold_gates` parameter is not applicable and will be ignored if set.

## Extrapolation Methods

After running circuits at each scale factor, the expectation values are extrapolated to zero noise. FiQCI EMS provides four extrapolation methods:

- **Exponential**: Fits an exponential decay model and works well when noise causes exponential decay of expectation values, which is common for depolarizing noise.

- **Richardson**: Uses Lagrange interpolation to compute exact coefficients that combine the measured values into a zero-noise estimate. This is a model-free method that makes no assumptions about the noise shape.

- **Polynomial**: Fits a polynomial of a given degree to the data. The degree defaults to `min(n_scales - 1, 2)` and can be set with the `extrapolation_degree` parameter.

- **Linear**: A special case of polynomial extrapolation with degree 1. Fits a straight line through the data points.

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
| `enabled` | `bool` | — | Enable or disable ZNE. |
| `fold_gates` | `list[str] \| None` | `None` | Gate names to fold (local folding only). `None` folds all two-qubit gates. |
| `scale_factors` | `list[int]` | `[1, 3, 5]` | Positive odd integers specifying the noise scale levels. At least two are required. |
| `folding_method` | `str` | `"local"` | `"local"` or `"global"`. |
| `extrapolation_method` | `str` | `"exponential"` | `"exponential"`, `"richardson"`, `"polynomial"`, or `"linear"`. |
| `extrapolation_degree` | `int \| None` | `None` | Polynomial degree (only for `"polynomial"` extrapolation). |

## Examples

- [Zero Noise Extrapolation Example](../notebooks/zero_noise_extrapolation_example) — runnable notebook demonstrating ZNE with default and custom settings.

## References

- Temme, K., Bravyi, S., Gambetta, J. M., "Error Mitigation for Short-Depth Quantum Circuits", [https://arxiv.org/abs/1612.02058](https://arxiv.org/abs/1612.02058)
- Li, Y., Benjamin, S. C., "Efficient Variational Quantum Simulator Incorporating Active Error Minimization", [https://arxiv.org/abs/1611.09301](https://arxiv.org/abs/1611.09301)
- Pegah Mohammadipour, Xiantao Li., "Direct Analysis of Zero-Noise Extrapolation: Polynomial Methods, Error Bounds, and Simultaneous Physical-Algorithmic Error Mitigation" [https://arxiv.org/abs/2502.20673](https://arxiv.org/abs/2502.20673)
