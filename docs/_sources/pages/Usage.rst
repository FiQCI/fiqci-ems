Usage
=========

FiQCI Error Mitigation Service (EMS) is a Python library for quantum error mitigation as part of the `Finnish Quantum Computing Infrastructure (FiQCI) <https://fiqci.fi>`_. It wraps IQM quantum backends and applies error mitigation, allowing users to run circuits with improved accuracy by specifying a mitigation level.

This python package can be pre-installed on a HPC system or installed by the user. The main goal of the project is to allow users using FiQCI quantum computers to easily add flags to run error mitigated quantum jobs.

Usage
-----

Start by initialising your IQM backend and a quantum circuit.

.. code:: python

   from iqm.qiskit_iqm import IQMProvider
   from qiskit import QuantumCircuit, transpile

   # Initialise backend
   provider = IQMProvider()
   backend = provider.get_backend()

   # Define a quantum circuit
   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure_all()

   # Transpile the circuit
   qc_transpiled = transpile(qc, backend=backend, initial_layout=qubit_indices)


For executing quantum jobs EMS provides three interfaces depending on your use case.

.. _usage-sampler:
.. _usage-estimator:
.. _usage-backend:

.. tab-set::

   .. tab-item:: Sampler

      .. rubric:: FiQCISampler - sampling interface

      For users who need measurement counts with built-in mitigation:

      .. code-block:: python

        from fiqci.ems import FiQCISampler

        # Using mitigation_level
        sampler = FiQCISampler(backend, mitigation_level=1)

        # Execute the job
        job = sampler.run(qc_transpiled, shots=2048)

        # Get results
        result = job.result()

        # Or manually set mitigation options
        sampler.rem(enabled=True, calibration_shots=2000, calibration_file="cals.json")

        # See applied and available options
        sampler.mitigation_options
    
      For more information on the sampler interface, see :doc:`FiQCISamplerUsage`.

   .. tab-item:: Estimator

      .. rubric:: FiQCIEstimator - expectation values

      Computes expectation values of Pauli observables directly from circuits:

      .. code-block:: python

        from fiqci.ems import FiQCIEstimator
        from qiskit.quantum_info import SparsePauliOp

        # Using mitigation_level
        estimator = FiQCIEstimator(backend, mitigation_level=1)

        # Define observables
        observables = SparsePauliOp.from_list([("ZZ", 1), ("IX", 1)])

        # Map observables to transpiled layout
        device_observables = observables.apply_layout(qc_transpiled.layout)

        # Execute the job
        job_collection = estimator.run(qc_transpiled, observables=device_observables, shots=2048)

        # Get expectation values
        evs = job_collection.expectation_values()

        # Access all jobs executed by estimator
        jobs = job_collection.jobs()

        # Or manually set mitigation options
        estimator.rem(enabled=True, calibration_shots=2000, calibration_file="cals.json")

        # See applied and available options
        estimator.mitigation_options

      For more information on the estimator interface, see :doc:`FiQCIEstimatorUsage`.

   .. tab-item:: Backend

      .. rubric:: FiQCIBackend - drop-in backend replacement

      FiQCIBackend is used under the hood by both sampler and estimator. Wraps any IQM backend and applies error mitigation to ``run()`` calls:

      .. code-block:: python

        from fiqci.ems import FiQCIBackend

        # Using mitigation_level
        backend = FiQCIBackend(backend, mitigation_level=1)

        # Execute the job
        job = backend.run(circuit, shots=1024)

        # Get the results
        result = job.result()

        # Or manually set mitigation options
        backend.rem(enabled=True, calibration_shots=2000, calibration_file="cals.json")

        # See applied and available options
        backend.mitigation_options

      Access raw (pre-mitigation) counts via ``backend.raw_counts``.

      The mitigation options for FiQCIBackend are the same as for FiQCISampler. For more information, see :doc:`FiQCISamplerUsage`.

Advanced Usage
~~~~~~~~~~~~~~

It is also possible to manually configure and directly use the M3 mitigator without the wrapper classes above. See :doc:`Examples` for details.