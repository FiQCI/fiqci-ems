from __future__ import annotations

import logging
from typing import Any, NamedTuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.result import Result

from fiqci.ems.utils import probabilities_to_counts

logger: logging.Logger = logging.getLogger(__name__)


class _KeyLayout(NamedTuple):
	"""Structure of result count keys, used to reduce them for M3 and restore them afterwards.

	Attributes:
		num_clbits: Total number of classical bits (length of a key with spaces removed).
		measured: Classical bit indices that are actually measured; all others are always zero.
		space_positions: Indices (in the original spaced key) at which spaces occur.
	"""

	num_clbits: int
	measured: frozenset[int]
	space_positions: tuple[int, ...]


def _key_layout(counts: dict[str, int], mapping: dict[int, int]) -> _KeyLayout:
	"""Describe the structure of count dictionary keys for M3 correction.

	M3 expects a bitstring with exactly one bit per measured qubit and no spaces. Result
	keys, however, contain a bit for every classical bit in the circuit, including spaces
	between classical registers (e.g. ``"001 00"``) and zero-filled bits for classical
	registers that are never measured. This computes the information needed to strip those
	extra bits before correction and restore them afterwards.

	Bitstrings are little-endian on the classical bit index: the character at spaceless
	index ``i`` corresponds to classical bit ``num_clbits - 1 - i`` (Qiskit MSB-left). A
	classical bit is "measured" iff it is a key of ``mapping`` (from
	``final_measurement_mapping``); every other bit is always zero in the results.

	Args:
	    counts: Count dictionary whose keys may contain spaces and unmeasured bits.
	    mapping: ``{classical_bit: qubit}`` mapping for the circuit's final measurements.

	Returns:
	    A :class:`_KeyLayout` describing the total bit count, the measured bits, and the
	    space positions of the original keys.
	"""
	sample_key = next(iter(counts))
	space_positions = tuple(i for i, char in enumerate(sample_key) if char == " ")
	num_clbits = len(sample_key) - len(space_positions)
	return _KeyLayout(num_clbits=num_clbits, measured=frozenset(mapping), space_positions=space_positions)


def _reduce_counts(counts: dict[str, int], layout: _KeyLayout) -> dict[str, int]:
	"""Strip spaces and unmeasured (always-zero) bits from count keys for M3 correction.

	Keys differing only in unmeasured bits collapse to the same reduced key (those bits are
	always zero), so their values are summed; in practice no collision occurs.

	Args:
	    counts: Count dictionary with full, spaced keys.
	    layout: Layout describing the key structure, from :meth:`_key_layout`.

	Returns:
	    Count dictionary keyed by measured bits only, with no spaces.
	"""
	reduced: dict[str, int] = {}
	for key, value in counts.items():
		spaceless = key.replace(" ", "")
		measured_bits = "".join(
			char for i, char in enumerate(spaceless) if (layout.num_clbits - 1 - i) in layout.measured
		)
		reduced[measured_bits] = reduced.get(measured_bits, 0) + value
	return reduced


def _expand_key(key: str, layout: _KeyLayout) -> str:
	"""Restore unmeasured zero bits and register spaces to a single reduced key."""
	measured_iter = iter(key)
	chars = [
		next(measured_iter) if (layout.num_clbits - 1 - i) in layout.measured else "0" for i in range(layout.num_clbits)
	]
	full = "".join(chars)
	for pos in layout.space_positions:
		full = full[:pos] + " " + full[pos:]
	return full


def _expand_counts(counts: dict[str, int], layout: _KeyLayout) -> dict[str, int]:
	"""Restore unmeasured zero bits and register spaces to reduced count keys.

	Inverse of :meth:`_reduce_counts`: each measured bit is placed back at its original
	position, unmeasured bits are filled with ``"0"``, and spaces are reinserted.

	Args:
	    counts: Count dictionary keyed by measured bits only (e.g. M3 output).
	    layout: Layout describing the key structure, from :meth:`_key_layout`.

	Returns:
	    Count dictionary with full, spaced keys matching the original structure.
	"""
	expanded: dict[str, int] = {}
	for key, value in counts.items():
		full = _expand_key(key, layout)
		expanded[full] = expanded.get(full, 0) + value
	return expanded


def _expand_quasi_probabilities(quasi_probabilities: dict[str, float], layout: _KeyLayout) -> dict[str, float]:
	"""Restore unmeasured zero bits and register spaces to reduced quasi-probability keys.

	Same key mapping as :meth:`_expand_counts`, but keeping the (possibly negative) float values M3
	produces instead of integer counts.
	"""
	expanded: dict[str, float] = {}
	for key, value in quasi_probabilities.items():
		full = _expand_key(key, layout)
		expanded[full] = expanded.get(full, 0.0) + float(value)
	return expanded


def _average_counts(counts_list: list[dict[str, int]]) -> dict[str, int]:
	"""Average multiple count dictionaries.

	Args:
	    counts_list: List of count dictionaries to average.

	Returns:
	    Averaged count dictionary with integer values. Apportioned by largest remainder, so it totals
	    the average shot count exactly.
	"""
	if len(counts_list) == 1:
		return counts_list[0]

	totals: dict[str, float] = {}
	for counts in counts_list:
		for key, value in counts.items():
			totals[key] = totals.get(key, 0.0) + value
	n = len(counts_list)
	shots = round(sum(totals.values()) / n)
	if shots == 0:
		return dict.fromkeys(totals, 0)
	return probabilities_to_counts([{key: value / (n * shots) for key, value in totals.items()}], shots)[0]


def _average_distributions(distributions: list[dict[str, float]]) -> dict[str, float]:
	"""Average quasi-probability distributions, keeping float (possibly negative) values.

	The counterpart of :meth:`_average_counts` for the unprojected M3 output, which must not be
	rounded to integers.

	Args:
	    distributions: List of quasi-probability dictionaries to average.

	Returns:
	    Averaged quasi-probability dictionary.
	"""
	if len(distributions) == 1:
		return dict(distributions[0])

	totals: dict[str, float] = {}
	for distribution in distributions:
		for key, value in distribution.items():
			totals[key] = totals.get(key, 0.0) + value
	n = len(distributions)
	return {key: value / n for key, value in totals.items()}


def _hex_key(key: str) -> str:
	"""Qiskit's hex form of a count or memory key given as a (possibly spaced) bitstring."""
	if key.startswith("0x"):
		return key
	bits = key.replace(" ", "")
	return hex(int(bits, 2)) if bits else "0x0"


def _fits_circuit(counts: dict[str, int], circuit: QuantumCircuit) -> bool:
	"""True when every key is hex or a bitstring with one bit per classical bit of ``circuit``."""
	num_clbits = getattr(circuit, "num_clbits", None)
	if not isinstance(num_clbits, int):
		return False
	for key in counts:
		if key.startswith("0x"):
			continue
		bits = key.replace(" ", "")
		if len(bits) != num_clbits or set(bits) - {"0", "1"}:
			return False
	return True


def _memory_from_counts(counts: dict[str, int], rng: np.random.Generator) -> list[str]:
	"""One hex entry per counted shot, in random order, since counts carry no shot order."""
	memory = [_hex_key(key) for key, value in counts.items() for _ in range(value)]
	rng.shuffle(memory)
	return memory


def _store_counts(
	entry: dict[str, Any], circuit: QuantumCircuit, counts: dict[str, int], memory: list[str] | None
) -> None:
	"""Store an experiment's counts and memory as Qiskit backends do.

	Keys are hex and the header describes the classical registers, so ``Result.get_counts()`` and
	``Result.get_memory()`` pad and split them by register, and ``BackendSamplerV2`` (which parses
	memory as hex) reads the right bits.

	Args:
	    entry: One experiment of a ``Result.to_dict()``, modified in place.
	    circuit: The circuit the experiment ran, which defines the classical registers.
	    counts: Counts keyed by spaced bitstrings or hex. Zero counts are dropped.
	    memory: Per-shot memory keyed the same way, or ``None`` to store none.

	Counts whose keys do not have one bit per classical bit of ``circuit`` are stored as given, with
	the header untouched, since they cannot be laid out by register.
	"""
	data = entry.setdefault("data", {})
	if not _fits_circuit(counts, circuit):
		data["counts"] = counts
		if memory is not None:
			data["memory"] = memory
		return
	hex_counts: dict[str, int] = {}
	for key, value in counts.items():
		# Averaging can leave outcomes at zero; backends report only observed outcomes.
		if value:
			hex_key = _hex_key(key)
			hex_counts[hex_key] = hex_counts.get(hex_key, 0) + value
	data["counts"] = hex_counts
	if memory is not None:
		data["memory"] = [_hex_key(shot) for shot in memory]

	header = entry.get("header")
	if not isinstance(header, dict):
		header = {}
		entry["header"] = header
	header["memory_slots"] = circuit.num_clbits
	creg_sizes = [[creg.name, creg.size] for creg in circuit.cregs]
	# Clbits outside any register cannot be described per register, so leave the keys unsplit.
	if sum(size for _, size in creg_sizes) == circuit.num_clbits:
		header["creg_sizes"] = creg_sizes
	else:
		header.pop("creg_sizes", None)


def _trim_result_to_groups(result: Result, num_groups: int) -> Result:
	"""Trim a flat (per-twirl) Result down to one entry per twirl group.

	The flat result list is ``[g0_orig, g0_tw1..g0_twN, g1_orig, g1_tw1.., ...]`` of length
	``num_groups * twirl_group_size``. We must keep the *representative* entry of each group
	(its original circuit, at stride ``twirl_group_size``) rather than the first ``num_groups``
	flat entries: a plain ``[:num_groups]`` slice keeps mostly group 0's twirl copies, so every
	kept entry carries group 0's header. Since ``Result.get_counts()`` reconstructs each
	bitstring's structure (creg sizes, memory slots) from the per-entry header, wrong headers
	yield misaligned, wrongly-structured counts even though ``_create_mitigated_result`` later
	overwrites ``data["counts"]``. Keeping ``results_list[i * stride]`` makes each kept entry's
	header match input circuit ``i``.

	Args:
	    result: Original Result object (flat, one entry per submitted twirl circuit).
	    num_groups: Number of twirl groups (== number of input circuits).

	Returns:
	    New Result object with one representative entry per group, in input-circuit order.
	"""
	from qiskit.result import Result as QiskitResult

	result_data = result.to_dict()
	results_list = result_data.get("results")
	if results_list is not None and num_groups:
		stride = len(results_list) // num_groups  # == twirl_group_size
		result_data["results"] = [results_list[i * stride] for i in range(num_groups)]
	return QiskitResult.from_dict(result_data)
