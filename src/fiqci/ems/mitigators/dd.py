"""
Functions for dynamical decoupling.
"""

from iqm.iqm_client import CircuitCompilationOptions, DDMode, DDStrategy
from typing import TypeAlias

PRXSequence: TypeAlias = list[tuple[float, float]]
DDGateSequenceEntry = list[tuple[int, str | PRXSequence, str]]


def build_dd_options(gate_sequences: list[DDGateSequenceEntry]) -> CircuitCompilationOptions:
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

    return CircuitCompilationOptions(
        dd_mode=DDMode.ENABLED,
        dd_strategy=DDStrategy(gate_sequences=gate_sequences),
    )
