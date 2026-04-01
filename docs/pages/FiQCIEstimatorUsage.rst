FiQCIEstimator
==============

:class:`~fiqci.ems.FiQCIEstimator` computes expectation values of Pauli observables from quantum circuits with built-in error mitigation. It supports both readout error mitigation (M3) and zero-noise extrapolation (ZNE).

Basic Configuration
-------------------

Initialize the estimator with an IQM backend, mitigation level, and optional parameters:

.. code-block:: python

   from fiqci.ems import FiQCIEstimator

   # Initialize estimator with mitigation level 1
   estimator = FiQCIEstimator(backend, mitigation_level=1, calibration_shots=2000, calibration_file="cals.json")

For more details see the API reference documentation for :class:`FiQCIEstimator`.

Mitigation Levels
-----------------

Mitigation levels apply predefined sets of error mitigation techniques.

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
     - Level 1 + Dynamical Decoupling
     - Dynamical Decoupling standard sequence (see :ref:`below <fiqci-estimator-dd>`)
   * - 3
     - Level 2 + Zero Noise Extrapolation
     - Exponential Extrapolation, Local Folding

Mitigation Options
------------------

Mitigators can also be configured manually using the provided methods.

- :ref:`Readout Error Mitigation (REM) <fiqci-estimator-rem>`
- :ref:`Zero Noise Extrapolation (ZNE) <fiqci-estimator-zne>`
- :ref:`Dynamical Decoupling (DD) <fiqci-estimator-dd>`

.. _fiqci-estimator-rem:

REM (Readout Error Mitigation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Readout error mitigation uses M3 (matrix-free measurement mitigation) to correct measurement errors. It is enabled by default at mitigation level 1.

Configure REM using the :meth:`~fiqci.ems.FiQCIEstimator.rem` method:

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


.. _fiqci-estimator-zne:

ZNE (Zero-Noise Extrapolation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZNE artificially scales circuit noise by folding gates, then extrapolates to the zero-noise limit. It is enabled at mitigation level 3.

Configure ZNE using the :meth:`~fiqci.ems.FiQCIEstimator.zne` method:

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

.. _fiqci-estimator-dd:

Dynamical Decoupling (DD)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamical decoupling inserts sequences of gates to mitigate decoherence. It is enabled at mitigation level 2.

Configure DD using the :meth:`~fiqci.ems.FiQCIEstimator.dd` method:

.. code-block:: python

   estimator.dd(enabled=True, gate_sequences=None) # None uses a standard set of sequences

The standard sequence is:

.. code-block:: python

   [
       (9, 'XYXYYXYX', 'asap'),
       (5, 'YXYX', 'asap'),
       (2, 'XX', 'center'),
   ]

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enabled``
     - ``True``
     - Enable or disable DD
   * - ``gate_sequences``
     - ``None``
     - List of (treshold_length, sequence, strategy) tuples defining DD behavior. If ``None``, uses a standard set of sequences.
         - ``treshold_length``: Minimum idle period (``treshold_length`` times duration of a single-qubit gate) to apply the sequence. If ``None``, uses ``len(sequence)`` or 2 if sequence is ``None``.
         - ``sequence``: List of gate names or :class:`~fiqci.ems.primitives.prx_sequence.PRXSequence` defining the DD sequence.
         - ``strategy``: Strategy for applying the sequence. One of:
             - ``"asap"``: Apply the sequence as soon as possible whenever the idle period exceeds the threshold.
             - ``"alap"``: Apply the sequence as late as possible whenever the idle period exceeds the threshold.
             - ``"center"``: Apply the sequence centered within idle periods exceeding the threshold.

Inspecting Options
------------------

Use the :attr:`~fiqci.ems.FiQCIEstimator.mitigator_options` property to view currently applied mitigation settings:

.. code-block:: python

   estimator.mitigator_options

Results
-------

:meth:`~fiqci.ems.FiQCIEstimator.run` returns a :class:`~fiqci.ems.primitives.fiqci_estimator.FiQCIEstimatorJobCollection` with the following methods:

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
