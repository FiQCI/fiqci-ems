# Dynamical Decoupling with FiQCI EMS

## What is Dynamical Decoupling:

Dynamical decoupling (DD) is a quantum error suppression technique that reduces the effects of decoherence on idle qubits. When a qubit is idle during a quantum circuit, e.g. while other qubits are being operated on, it is exposed to unwanted interactions with its environment. These interactions cause the qubit's state to decay over time, introducing errors into the computation.

DD works by inserting sequences of fast control pulses (such as X and Y gates) on idle qubits. These pulses effectively refocus the qubit's evolution, averaging out the environmental noise over time. The key idea is that if noise acts slowly compared to the rate of the applied pulses, the net effect of the noise is suppressed.

Common DD sequences include:

- **XY**: Alternating X and Y gates. A simple and widely used sequence that protects against both bit-flip and phase-flip noise.
- **XYXY**: A repeated XY sequence offering stronger suppression for longer idle periods.
- **Custom PRX sequences**: User-defined rotation sequences for advanced noise profiles.

## Usage

Dynamical decoupling is enabled by default at ``mitigation_level=2`` or it can be enabled with the primitives `dd()` method.

For more detailed usage see [FiQCISampler](FiQCISamplerUsage.rst) or [FiQCIEstimator](FiQCIEstimatorUsage.rst)

### Combining DD with other compilation options

DD is submitted through IQM's `CircuitCompilationOptions`, which is also how you reach settings EMS
does not wrap (heralding, MOVE gate validation, MOVE gate frame tracking). Passing your own options to
`run()` keeps every non-DD field and only sets the DD ones, so the two compose:

```python
from iqm.iqm_client import CircuitCompilationOptions, MoveGateValidationMode

sampler = FiQCISampler(backend, mitigation_level=2)  # DD on
sampler.run(
    circuit,
    shots=4096,
    circuit_compilation_options=CircuitCompilationOptions(
        move_gate_validation=MoveGateValidationMode.ALLOW_PRX
    ),
)
```

The DD strategy from `dd()` (or the mitigation level) wins over a `dd_mode`/`dd_strategy` set in your
own options, and you get a warning if both are configured. To submit with your own DD strategy
instead, turn EMS's off with `dd(enabled=False)`.

## References

- [IQM Academy: Dynamical Decoupling](https://www.iqmacademy.com/learn/errorreduction/02-dd/)
- Ezzell, N., Pokharel, B., Tewala, L., Quiroz, G., & Lidar, D. A. (2023). Dynamical decoupling for superconducting qubits: A performance survey [https://arxiv.org/abs/2207.03670](https://arxiv.org/abs/2207.03670)
