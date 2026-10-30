"""
Functions for dynamical decoupling.
"""

import dataclasses

from iqm.iqm_client import CircuitCompilationOptions, DDMode, DDStrategy
from typing import TypeAlias

PRXSequence: TypeAlias = list[tuple[float, float]]
DDGateSequenceEntry = tuple[int, str | PRXSequence, str]


def build_dd_options(
	gate_sequences: list[DDGateSequenceEntry], base: CircuitCompilationOptions | None = None
) -> CircuitCompilationOptions:
	"""
	Build compilation options for dynamical decoupling.

	Args:
	    gate_sequences: List of (threshold_length, sequence, strategy) tuples defining DD behavior.
	        - threshold_length: Length of idle time before applying DD. Defaults to sequence length or 2.
	        - sequence: DD sequence as a string (e.g., "XYXY") or list of rotation angle tuples. Defaults to "XY".
	        - strategy: "asap", "alap", or "center". Defaults to "asap".
	    base: Compilation options to add DD to, keeping every non-DD field (heralding, MOVE gate
	        validation and frame tracking, ...) as given. Defaults to IQM's defaults.

	Returns:
	    CircuitCompilationOptions with the specified DD settings.
	"""
	dd_strategy = DDStrategy(gate_sequences=gate_sequences)
	if base is not None:
		return dataclasses.replace(base, dd_mode=DDMode.ENABLED, dd_strategy=dd_strategy)

	return CircuitCompilationOptions(dd_mode=DDMode.ENABLED, dd_strategy=dd_strategy)
