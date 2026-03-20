"""Unit tests for FiQCISampler class."""

from unittest.mock import Mock, patch

import pytest
from qiskit import QuantumCircuit

from fiqci.ems.fiqci_sampler import FiQCISampler


class TestFiQCISampler:
	"""Tests for FiQCISampler class."""

	@pytest.fixture
	def mock_backend(self) -> Mock:
		"""Create a mock IQM backend."""
		backend = Mock()
		backend.name = "MockBackend"
		backend.num_qubits = 5
		return backend

	@pytest.fixture
	def mock_circuit(self) -> QuantumCircuit:
		"""Create a simple quantum circuit."""
		qc = QuantumCircuit(2)
		qc.h(0)
		qc.cx(0, 1)
		qc.measure_all()
		return qc

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_init_creates_fiqci_backend(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that FiQCISampler creates a FiQCIBackend on init."""
		sampler = FiQCISampler(mock_backend, mitigation_level=1, calibration_shots=2000)
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 2000, None)

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_init_default_parameters(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that default parameters are passed to FiQCIBackend."""
		sampler = FiQCISampler(mock_backend)
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 1000, None)

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_init_with_calibration_files(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that calibration_files parameter is forwarded."""
		sampler = FiQCISampler(mock_backend, calibration_files="cal.json")
		mock_fiqci_backend_class.assert_called_once_with(mock_backend, 1, 1000, "cal.json")

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_run_delegates_to_backend(self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that run() delegates to FiQCIBackend.run()."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.run.return_value = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		sampler.run(mock_circuit, shots=2048)

		mock_fiqci_backend.run.assert_called_once_with(mock_circuit, shots=2048)

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_run_default_shots(self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that _run uses default 2048 shots when not specified."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.run.return_value = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		sampler._run(mock_circuit)

		mock_fiqci_backend.run.assert_called_once_with(mock_circuit, shots=2048)

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_run_passes_kwargs(self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that run() passes extra keyword arguments through."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.run.return_value = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		sampler.run(mock_circuit, shots=512, some_option="value")

		mock_fiqci_backend.run.assert_called_once_with(mock_circuit, shots=512, some_option="value")

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_run_returns_backend_result(self, mock_fiqci_backend_class: Mock, mock_backend: Mock, mock_circuit: QuantumCircuit) -> None:
		"""Test that run() returns the result from FiQCIBackend.run()."""
		mock_fiqci_backend = Mock()
		expected_result = Mock()
		mock_fiqci_backend.run.return_value = expected_result
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		result = sampler.run(mock_circuit)

		assert result is expected_result

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_run_with_circuit_list(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that run() works with a list of circuits."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend.run.return_value = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		circuits = [QuantumCircuit(2), QuantumCircuit(2)]
		sampler = FiQCISampler(mock_backend)
		sampler.run(circuits, shots=1024)

		mock_fiqci_backend.run.assert_called_once_with(circuits, shots=1024)

	@patch("fiqci.ems.fiqci_sampler.FiQCIBackend")
	def test_backend_attribute_is_fiqci_backend(self, mock_fiqci_backend_class: Mock, mock_backend: Mock) -> None:
		"""Test that the backend attribute is the FiQCIBackend instance."""
		mock_fiqci_backend = Mock()
		mock_fiqci_backend_class.return_value = mock_fiqci_backend

		sampler = FiQCISampler(mock_backend)
		assert sampler.backend is mock_fiqci_backend
