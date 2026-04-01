"""
Functions for dynamical decoupling.
"""

from iqm.iqm_client import CircuitCompilationOptions, DDMode, DDStrategy
from typing import TypeAlias

PRXSequence: TypeAlias = list[tuple[float, float]]
DDGateSequenceEntry = list[tuple[int, str | PRXSequence, str]]


def build_dd_options(gate_sequences: list[DDGateSequenceEntry] | None = None) -> CircuitCompilationOptions:
    """
    Build compilation options for dynamical decoupling.

    Args:
        gate_sequences: List of (treshold_length, sequence, strategy) tuples defining DD behavior.
            - treshold_length: Length of idle time before applying DD. Defaults to sequence length or 2.
            - sequence: DD sequence as a string (e.g., "XYXY") or list of rotation angle tuples. Defaults to "XY".
            - strategy: "asap", "alap", or "center". Defaults to "asap".

    Returns:
        CircuitCompilationOptions with the specified DD settings.
    """

    resolved = []
    for treshold_length, sequence, strategy in gate_sequences:
        if treshold_length is None and sequence is not None:
            treshold_length = len(sequence)
        elif treshold_length is None:
            treshold_length = 2

        if strategy is None:
            strategy = "asap"

        if sequence is None:
            sequence = "XY"

        resolved.append((treshold_length, sequence, strategy))

    return CircuitCompilationOptions(
        dd_mode=DDMode.ENABLED,
        dd_strategy=DDStrategy(gate_sequences=resolved),
    )
