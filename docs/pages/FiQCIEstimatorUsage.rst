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

Transpile the circuit with :func:`~iqm.qiskit_iqm.transpile_to_IQM` and map the observables onto the
transpiled circuit's layout. ``remove_final_rzs=False`` is required: the estimator measures in the X
and Y bases, and dropping the final RZ gates changes those expectation values.

.. code-block:: python

   from iqm.qiskit_iqm import transpile_to_IQM
   from qiskit.quantum_info import SparsePauliOp

   tr_qc = transpile_to_IQM(qc, backend, remove_final_rzs=False, optimization_level=3)

   observables = SparsePauliOp.from_list([("ZZ", 1), ("IX", 1)])
   device_observables = observables.apply_layout(tr_qc.layout)

The examples below use ``tr_qc`` and ``device_observables``.

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
- :ref:`Pauli Twirling <fiqci-estimator-pt>`

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
     - List of real numbers >= 1 for noise scaling. At least 2 required. Odd integers fold exactly; other values are approximated by partial/random folding. May be a list of lists (one per submitted circuit) to use different scale factors per circuit.
   * - ``folding_method``
     - ``"local"``
     - ``"local"`` (per-gate folding) or ``"global"`` (whole-circuit folding). When ``"global"``, ``fold_gates`` is ignored.
   * - ``extrapolation_method``
     - ``"exponential"``
     - Extrapolation fit method. One of: ``"exponential"``, ``"richardson"``, ``"polynomial"``, ``"linear"``, or a user-defined callable invoked as ``fn(expectation_values, scale_factors)`` (returning a list of floats). ``expectation_values`` has shape ``(n_scales, n_obs)`` and ``scale_factors`` is the achieved scale-factor list for the pair.
   * - ``extrapolation_degree``
     - ``None``
     - Polynomial degree (only for ``"polynomial"`` method). Defaults to ``min(n_scales - 1, 2)``.
   * - ``seed``
     - ``None``
     - Seed for the random gate sampling used to approximate non-odd-integer scale factors.

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
     - List of (threshold_length, sequence, strategy) tuples defining DD behavior. If ``None``, uses a standard set of sequences.
         - ``threshold_length``: Minimum idle period (``threshold_length`` times duration of a single-qubit gate) to apply the sequence. If ``None``, uses ``len(sequence)`` or 2 if sequence is ``None``.
         - ``sequence``: List of gate names or :class:`~fiqci.ems.primitives.prx_sequence.PRXSequence` defining the DD sequence.
         - ``strategy``: Strategy for applying the sequence. One of:
             - ``"asap"``: Apply the sequence as soon as possible whenever the idle period exceeds the threshold.
             - ``"alap"``: Apply the sequence as late as possible whenever the idle period exceeds the threshold.
             - ``"center"``: Apply the sequence centered within idle periods exceeding the threshold.

.. _fiqci-estimator-pt:

Pauli Twirling
~~~~~~~~~~~~~~~~

Pauli twirling sandwiches two-qubit gates with random single-qubit Pauli gates to mitigate coherent errors. It is enabled at mitigation level 3.

Configure Pauli Twirling using the :meth:`~fiqci.ems.FiQCIEstimator.pauli_twirl` method:

.. code-block:: python

   estimator.pauli_twirl(enabled=True, num_twirls=10, gates_to_twirl=None) # None twirls all two-qubit gates

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``enabled``
     - ``True``
     - Enable or disable Pauli twirling
   * - ``num_twirls``
     - ``10``
     - Number of twirled variant circuits to generate for each original circuit
   * - ``gates_to_twirl``
     - ``None``
     - List of two-qubit gates to twirl. If ``None``, twirls all two-qubit gates.

Inspecting Options
------------------

Use the :attr:`~fiqci.ems.FiQCIEstimator.mitigator_options` property to view currently applied mitigation settings:

.. code-block:: python

   estimator.mitigator_options

The returned dictionary is a **copy**, so mutating it does not reconfigure the estimator; use
:meth:`~fiqci.ems.FiQCIEstimator.zne` / :meth:`~fiqci.ems.FiQCIEstimator.rem` /
:meth:`~fiqci.ems.FiQCIEstimator.dd` / :meth:`~fiqci.ems.FiQCIEstimator.pauli_twirl` for that, which
validate their input. The live ``M3IQM`` mitigator under ``rem`` is shared by reference rather than
copied, since it owns the calibration data.

Counting Circuits
-----------------

Use :meth:`~fiqci.ems.FiQCIEstimator.total_circuits_generated` to see how many circuits will actually be executed under the current mitigation settings (accounts for measurement-basis splitting, ZNE scale factors and Pauli twirls):

.. code-block:: python

   estimator.total_circuits_generated(num_base_circuits=1, observables=device_observables, detailed=True)

Set ``detailed=True`` to print the breakdown and return a dictionary with each multiplier; otherwise the method returns the total as an ``int``. The count does not include REM calibration circuits.

Running Circuits
----------------

:meth:`~fiqci.ems.FiQCIEstimator.run` accepts a ``max_batch_size`` parameter (default ``100``) that controls how many circuits are sent to the backend in a single job. All measurement-basis subcircuits across every circuit/observable pair (and every ZNE scale factor) are flattened and split into batches of this size, then re-combined transparently.

.. code-block:: python

   job = estimator.run(tr_qc, observables=device_observables, shots=2048, max_batch_size=100)

``run`` returns **immediately** with a :class:`~fiqci.ems.primitives.fiqci_estimator.FiQCIEstimatorJob` handle without waiting for results. Polling the underlying job works right away (``job.status()``, ``job.job_ids()``); the expectation values are computed the first time you call ``expectation_values()`` / ``raw_expectation_values()``, and cached.

The handle also lets you inspect a multi-batch run before it finishes:

.. code-block:: python

   job.job_ids()         # backend job id of every batch (None for any unsubmitted batch)
   job.statuses()        # per-batch JobStatus, in submission order
   job.status()          # single aggregated status across all batches
   job.done()            # True once every batch has reached a terminal state
   job.partial_results() # per-batch results for batches that have already completed
   job.mitigator_options # frozen snapshot of the mitigation settings this run used

Unlike :attr:`~fiqci.ems.FiQCIEstimator.mitigator_options` on the estimator (which reflects the *current*, mutable settings), the handle's ``mitigator_options`` is a snapshot frozen at submission. It reports the ``zne`` configuration together with the underlying ``mitigation_level``, ``rem``, ``dd`` and ``pauli_twirl`` settings this run actually used, and never changes even if you reconfigure the estimator afterwards. (The per-pair scale factors are available separately via ``requested_scale_factors()`` / ``achieved_scale_factors()`` below.)


.. note::

   As with the sampler, submission is not atomic. If the backend rejects a circuit partway through, ``run`` logs a warning and still returns a handle (rejected/skipped batches report ``ERROR``/``CANCELLED``. To inspect use ``job.statuses()``). In that case ``expectation_values()`` and ``result()`` raise :class:`~fiqci.ems.BatchFailedError`, since the values cannot be computed without all circuits.

Results
-------

:meth:`~fiqci.ems.FiQCIEstimator.run` returns a :class:`~fiqci.ems.primitives.fiqci_estimator.FiQCIEstimatorJob` that wraps the single backend job produced by the run. Expectation values are computed lazily on first access and cached. It exposes the following methods:

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``expectation_values(index=None)``
     - Mitigated expectation values
   * - ``raw_expectation_values(index=None)``
     - Raw (pre-extrapolation) expectation values
   * - ``job()``
     - The underlying backend job
   * - ``result()``
     - Combined ``Result`` from the backend job
   * - ``observables(index=None)``
     - Observables used in the computation
   * - ``requested_scale_factors(index=None)``
     - ZNE scale factors requested for the run, one list per circuit/observable pair
   * - ``achieved_scale_factors(index=None)``
     - ZNE scale factors folding actually realised (the extrapolation x-axis), same shape
   * - ``mitigator_options``
     - Frozen snapshot of the mitigation settings this run used (``zne`` config plus the backend ``mitigation_level``/``rem``/``dd``/``pauli_twirl``)
   * - ``standard_errors(index=None)``
     - Standard errors of the expectation values (shot noise and, when ZNE is enabled, the extrapolation uncertainty)

Standard Errors
~~~~~~~~~~~~~~~

:meth:`~fiqci.ems.primitives.fiqci_estimator.FiQCIEstimatorJob.standard_errors` reports the uncertainty of each expectation value, computed lazily and cached alongside the values. It mirrors the shape of ``expectation_values``: one entry per circuit/observable pair, each a dict of per-Pauli-term standard errors.

.. code-block:: python

   job = estimator.run(tr_qc, observables=device_observables, shots=2048)

   job.expectation_values(0)   # e.g. [0.92, -0.01, ...]
   job.standard_errors(0)      # {"shot_error": [...], "zne_extrapolation_error": ..., "total": [...]}

.. list-table::
   :header-rows: 1

   * - Key
     - Description
   * - ``shot_error``
     - Statistical standard error of the raw measurement, :math:`\sqrt{(1 - \langle P \rangle^2) / N}` per term. When ZNE is enabled this is taken at the unfolded (scale 1) point. This matches the convention used by Qiskit's sampling-based ``EstimatorV2``.
   * - ``zne_extrapolation_error``
     - Standard error of the extrapolated value: the per-scale shot errors propagated through the (linear) extrapolator. ``None`` when ZNE is disabled, or when a user-defined extrapolation callable reports no standard errors. See :ref:`ZNE <fiqci-estimator-zne>`.
   * - ``total``
     - Standard error of the value :meth:`~fiqci.ems.primitives.fiqci_estimator.FiQCIEstimatorJob.expectation_values` actually returns — ``shot_error`` when ZNE is off, ``zne_extrapolation_error`` when ZNE is on. Not a quadrature sum, since the extrapolation error already incorporates the shot noise.

.. note::

   Only statistical error is reported. The estimator does not account for ZNE *model* bias (the systematic error from the chosen extrapolation shape), nor does it inflate ``shot_error`` by the M3 readout-mitigation overhead at mitigation level ≥ 1.

Examples
--------

- :doc:`Using The FiQCI Estimator <../notebooks/expectation_values_fiqci_estimator>`
- :doc:`Using Zero Noise Extrapolation <../notebooks/zero_noise_extrapolation_example>`
