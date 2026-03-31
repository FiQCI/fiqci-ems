FiQCIEstimator
==============

``FiQCIEstimator`` computes expectation values of Pauli observables from quantum circuits with built-in error mitigation. It supports both readout error mitigation (M3) and zero-noise extrapolation (ZNE).

For general usage instructions, see :ref:`Usage <usage-estimator>`.

Mitigation Options
------------------

REM (Readout Error Mitigation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Readout error mitigation uses M3 (matrix-free measurement mitigation) to correct measurement errors. It is enabled by default at mitigation level 1.

Configure REM using the ``rem()`` method:

.. code-block:: python

   estimator.rem(enabled=True, calibration_shots=2000, calibration_file="cals.json")

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enabled``
     - ``True``
     - Enable or disable readout error mitigation
   * - ``calibration_shots``
     - ``1000``
     - Number of shots used for M3 calibration circuits
   * - ``calibration_file``
     - ``None``
     - Path to save/load calibration data (JSON). Reuses cached calibrations when available.

ZNE (Zero-Noise Extrapolation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZNE artificially scales circuit noise by folding gates, then extrapolates to the zero-noise limit. It is enabled at mitigation level 3.

Configure ZNE using the ``zne()`` method:

.. code-block:: python

   estimator.zne(
       enabled=True,
       fold_gates=["cx", "cz"],
       scale_factors=[1, 3, 5],
       folding_method="local",
       extrapolation_method="exponential",
   )

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enabled``
     - ``True``
     - Enable or disable ZNE
   * - ``fold_gates``
     - ``None``
     - Gate names to fold (e.g. ``["cx", "cz"]``). If ``None``, folds all gates.
   * - ``scale_factors``
     - ``[1, 3, 5]``
     - List of positive odd integers for noise scaling. At least 2 required.
   * - ``folding_method``
     - ``"local"``
     - ``"local"`` (per-gate folding) or ``"global"`` (whole-circuit folding). When ``"global"``, ``fold_gates`` is ignored.
   * - ``extrapolation_method``
     - ``"exponential"``
     - Extrapolation fit method. One of: ``"exponential"``, ``"richardson"``, ``"polynomial"``, ``"linear"``.
   * - ``extrapolation_degree``
     - ``None``
     - Polynomial degree (only for ``"polynomial"`` method). Defaults to ``min(n_scales - 1, 2)``.

Mitigation Levels
-----------------

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

Inspecting Options
------------------

Use the ``mitigator_options`` property to view currently applied mitigation settings:

.. code-block:: python

   estimator.mitigator_options

Results
-------

``FiQCIEstimator.run()`` returns a ``FiQCIEstimatorJobCollection`` with the following methods:

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``expectation_values(index=None)``
     - Mitigated expectation values
   * - ``raw_expectation_values(index=None)``
     - Raw (pre-extrapolation) expectation values
   * - ``jobs()``
     - All jobs executed by the estimator
   * - ``results()``
     - Results for each job
   * - ``observables(index=None)``
     - Observables used in the computation

Examples
--------

- :doc:`Using The FiQCI Estimator <../notebooks/expectation_values_fiqci_estimator>`
- :doc:`Using Zero Noise Extrapolation <../notebooks/zero_noise_extrapolation_example>`
