"""
Functions for dynamical decoupling.
"""

from iqm.iqm_client import CircuitCompilationOptions, DDMode, DDStrategy

def build_dd_options(treshold_length: int | None, sequence: str | list[tuple] | None, strategy:str | None) -> CircuitCompilationOptions:
    """
    Build compilation options for dynamical decoupling.

    Args:
        treshold_length: Length of idle time before applying DD. Sequences will be applied to idle qubits for idle times longer than this threshold.
        sequence: DD sequence to apply, either as a string (e.g., "XYXY") or a list of rotation angle tuples (e.g., [(np.pi/2, 0), (np.pi, np.pi/2)]).
        strategy: Strategy for applying the sequence.
                - "asap": As soon as possible after the idle time threshold is reached.
                - "alap": As late as possible before the next gate on the qubit.
                - "center": Centered within the idle time.

    Returns:
        CircuitCompilationOptions with the specified DD settings.
    """

    if treshold_length is None and sequence is None and strategy is None:
        # If no parameters are provided, enable DD with default settings.
        return CircuitCompilationOptions(dd_mode=DDMode.ENABLED)

    if treshold_length is None and sequence is not None:
        treshold_length = len(sequence)  # Default threshold length to the sequence length if not provided.
    
    elif treshold_length is None:
        treshold_length = 2  # Default threshold length if sequence is also not provided.

    if strategy is None:
        strategy = "asap"  # Default strategy.

    if sequence is None:
        sequence = "XY"  # Default sequence.

    return CircuitCompilationOptions(
        dd_mode=DDMode.ENABLED,
        dd_strategy=DDStrategy(gate_sequences=[(treshold_length, sequence, strategy)]),
    )
