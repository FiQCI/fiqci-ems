Usage
=========

FiQCI Error Mitigation Service (EMS) is a Python library for quantum error mitigation as part of the `Finnish Quantum Computing Infrastructure (FiQCI) <https://fiqci.fi>`_. It wraps IQM quantum backends and applies error mitigation, allowing users to run circuits with improved accuracy by specifying a mitigation level.

This python package can be pre-installed on a HPC system or installed by the user. The main goal of the project is to allow users using FiQCI quantum computers to easily add flags to run error mitigated quantum jobs.

Mitigation Levels
-----------------

.. tab-set::

   .. tab-item:: Sampler

      .. list-table::
         :header-rows: 1
         :align: center

         * - Level
           - Mitigation Applied
           - Technique
         * - 0
           - None
           - Raw results
         * - 1
           - Readout Error Mitigation
           - M3 (matrix-free measurement mitigation)
         * - 2
           - Level 1 + additional
           - TBD
         * - 3
           - Level 2 + additional
           - TBD

   .. tab-item:: Estimator

      .. list-table::
         :header-rows: 1
         :align: center

         * - Level
           - Mitigation Applied
           - Technique
         * - 0
           - None
           - Raw results
         * - 1
           - Readout Error Mitigation
           - M3 (matrix-free measurement mitigation)
         * - 2
           - Level 1 + additional
           - TBD
         * - 3
           - Level 2 + Zero Noise Extrapolation
           - Extrapolation

.. note::

   ``FiQCIBackend`` mitigation levels correspond to the Sampler levels.

The default is level 1, which applies M3 readout error mitigation.

Installation
------------

`Uv <https://docs.astral.sh/uv/getting-started/installation/>`_ is recommended for installation

.. code-block:: bash
    
    uv pip install fiqci-ems
    #or
    uv add fiqci-ems

Requires Python 3.11 or 3.12.

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

Advanced Usage
~~~~~~~~~~~~~~

It is also possible to manually configure and directly use the M3 mitigator without the wrapper classes above. See :doc:`Examples` for details.

Configuration
-------------

All three interfaces accept the same core options:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``mitigation_level``
     - ``1``
     - Mitigation level (0-3)
   * - ``calibration_shots``
     - ``1000``
     - Shots used for M3 calibration circuits
   * - ``calibration_file``
     - ``None``
     - Path to save/load calibration data (JSON)

Mitigation can also be configured directly. See the docs for ``FiQCISampler``, ``FiQCIEstimator``, and ``FiQCIBackend`` to see all available options.

Having trouble?
---------------

Contact `servicedesk@csc.fi <mailto:servicedesk@csc.fi>`_ or raise an issue here.
