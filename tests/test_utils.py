"""Tests for utility functions."""

import numpy as np

from fiqci.ems.utils import probabilities_to_counts


def test_probabilities_to_counts():
	"""Test probabilities_to_counts with various inputs."""
	# Test single dict
	probs = {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
	result = probabilities_to_counts(probs, 1000)
	assert isinstance(result, list)
	assert len(result) == 1
	assert all(isinstance(v, int) for v in result[0].values())

	# Test list of dicts
	probs_list = [{"00": 0.5, "11": 0.5}, {"00": 0.75, "11": 0.25}]
	result = probabilities_to_counts(probs_list, 100)
	assert len(result) == 2
	assert all(isinstance(v, int) for v in result[0].values())

	# Test numpy float types
	probs_numpy = {
		"10": np.float32(0.23291016),
		"11": np.float32(0.23828125),
		"00": np.float32(0.25146484),
		"01": np.float32(0.27734375),
	}
	result = probabilities_to_counts(probs_numpy, 2048)
	assert len(result) == 1
	assert all(isinstance(v, int) for v in result[0].values())
