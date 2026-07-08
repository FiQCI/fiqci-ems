"""Tests for the count-key helpers in ``fiqci.ems.backend.counts``.

These directly unit-test the reduce/expand/layout helpers that strip result count keys
down for M3 correction and restore them afterwards. The reduce/expand pair is otherwise
only exercised indirectly through the full M3 workflow in ``test_rem.py``.
"""

from qiskit.result import Result

from fiqci.ems.backend.counts import (
	_average_counts,
	_expand_counts,
	_key_layout,
	_reduce_counts,
	_trim_result_to_groups,
)


class TestKeyLayout:
	"""Tests for ``_key_layout``."""

	def test_no_spaces_all_measured(self):
		"""A single spaceless register with every bit measured."""
		layout = _key_layout({"101": 10}, {0: 8, 1: 16, 2: 24})
		assert layout.num_clbits == 3
		assert layout.measured == frozenset({0, 1, 2})
		assert layout.space_positions == ()

	def test_spaces_and_unmeasured_bits(self):
		"""Spaces are located and excluded from the classical-bit count."""
		# "001 00": one space at index 3, five classical bits, only bits 0 and 1 measured.
		layout = _key_layout({"001 00": 5}, {0: 8, 1: 16})
		assert layout.num_clbits == 5
		assert layout.measured == frozenset({0, 1})
		assert layout.space_positions == (3,)

	def test_multiple_spaces(self):
		"""Multiple register separators are all recorded."""
		layout = _key_layout({"0 1 0": 1}, {0: 0, 1: 1, 2: 2})
		assert layout.num_clbits == 3
		assert layout.space_positions == (1, 3)


class TestReduceCounts:
	"""Tests for ``_reduce_counts``."""

	def test_strips_spaces_and_unmeasured_bits(self):
		"""Only measured bits survive; spaces and unmeasured bits are dropped.

		Little-endian: spaceless index ``i`` maps to classical bit ``num_clbits - 1 - i``.
		With num_clbits=3 and measured={0, 2}, key "101" -> keep bits 2 and 0 -> "11".
		"""
		layout = _key_layout({"101": 1}, {0: 0, 2: 2})
		assert _reduce_counts({"101": 7}, layout) == {"11": 7}

	def test_reinsert_ignores_spaces(self):
		"""Spaced multi-register keys reduce to spaceless measured bits."""
		layout = _key_layout({"0 0": 1}, {0: 8, 1: 16})
		reduced = _reduce_counts({"0 0": 400, "1 1": 350, "0 1": 130, "1 0": 120}, layout)
		assert reduced == {"00": 400, "11": 350, "01": 130, "10": 120}

	def test_unmeasured_bits_collapse_and_sum(self):
		"""Keys differing only in an always-zero unmeasured bit collapse and their values sum."""
		# 3 clbits, only bit 0 measured. Bits differing in the (unmeasured) high bits reduce equally.
		layout = _key_layout({"000": 1}, {0: 0})
		reduced = _reduce_counts({"000": 10, "100": 20, "010": 5}, layout)
		# All three keep only bit 0 == "0", summing to 35.
		assert reduced == {"0": 35}


class TestExpandCounts:
	"""Tests for ``_expand_counts`` (inverse of ``_reduce_counts``)."""

	def test_fills_zeros_and_restores_spaces(self):
		"""Measured bits are placed back, unmeasured filled with 0, spaces reinserted."""
		layout = _key_layout({"0 0": 1}, {0: 8, 1: 16})
		expanded = _expand_counts({"00": 400, "11": 350, "01": 130}, layout)
		assert expanded == {"0 0": 400, "1 1": 350, "0 1": 130}

	def test_places_measured_bits_at_correct_positions(self):
		"""Unmeasured interior bit is zero-filled while measured bits keep their positions."""
		# 3 clbits, measured {0, 2}. Reduced "11" -> bit2=1, bit1=0 (unmeasured), bit0=1 -> "101".
		layout = _key_layout({"000": 1}, {0: 0, 2: 2})
		assert _expand_counts({"11": 9}, layout) == {"101": 9}

	def test_round_trip_identity_when_unmeasured_bits_zero(self):
		"""reduce -> expand restores the original keys when unmeasured bits are zero."""
		original = {"010 0": 100, "001 0": 50, "000 0": 25}
		# 4 clbits (space at index 3), measured bits 0,1,2 -> the leading bit (bit 3) is unmeasured/zero.
		layout = _key_layout(original, {0: 0, 1: 1, 2: 2})
		reduced = _reduce_counts(original, layout)
		assert _expand_counts(reduced, layout) == original


class TestAverageCounts:
	"""Tests for ``_average_counts``."""

	def test_single_returns_same_object(self):
		counts = {"00": 500, "11": 500}
		assert _average_counts([counts]) is counts

	def test_averages_and_rounds(self):
		result = _average_counts([{"00": 1, "11": 2}, {"00": 2, "11": 1}, {"00": 1, "11": 1}])
		# (1+2+1)/3 and (2+1+1)/3 both round to 1.
		assert result == {"00": 1, "11": 1}

	def test_non_overlapping_keys(self):
		result = _average_counts([{"00": 1000}, {"11": 1000}])
		assert result == {"00": 500, "11": 500}


class TestTrimResultToGroups:
	"""Tests for ``_trim_result_to_groups``."""

	@staticmethod
	def _make_result(names_and_counts: list[tuple[str, dict[str, int]]]) -> Result:
		return Result.from_dict(
			{
				"results": [
					{"data": {"counts": counts}, "header": {"name": name}, "shots": 1024, "success": True}
					for name, counts in names_and_counts
				],
				"backend_name": "mock",
				"job_id": "test",
				"qobj_id": "test",
				"success": True,
				"status": "COMPLETED",
			}
		)

	def test_keeps_group_representatives_at_stride(self):
		"""Keeps the original (stride-0) entry of each group, not the first N flat entries."""
		# Flat: [g0, g0_tw, g1, g1_tw] with twirl_group_size=2, 2 groups.
		result = self._make_result(
			[("g0", {"00": 500}), ("g0_tw", {"11": 500}), ("g1", {"01": 500}), ("g1_tw", {"10": 500})]
		)
		trimmed = _trim_result_to_groups(result, 2).to_dict()
		assert [r["header"]["name"] for r in trimmed["results"]] == ["g0", "g1"]
		assert [r["data"]["counts"] for r in trimmed["results"]] == [{"00": 500}, {"01": 500}]

	def test_no_twirling_is_identity(self):
		"""stride == 1 keeps every entry (num_groups == number of results)."""
		result = self._make_result([("c0", {"00": 1}), ("c1", {"11": 1})])
		trimmed = _trim_result_to_groups(result, 2).to_dict()
		assert [r["header"]["name"] for r in trimmed["results"]] == ["c0", "c1"]

	def test_zero_groups_returns_unchanged(self):
		"""num_groups == 0 leaves the result list untouched (guards against div-by-zero)."""
		result = self._make_result([("c0", {"00": 1})])
		trimmed = _trim_result_to_groups(result, 0).to_dict()
		assert len(trimmed["results"]) == 1
