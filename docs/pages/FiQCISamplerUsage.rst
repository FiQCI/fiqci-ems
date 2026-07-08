FiQCISampler
============

:class:`~fiqci.ems.FiQCISampler` is a sampling interface that wraps an IQM backend and applies error mitigation to measurement results. It executes quantum circuits and returns mitigated counts.


Basic Configuration
-------------------

Initialize the sampler with an IQM backend, mitigation level, and optional parameters:

.. code-block:: python

   from fiqci.ems import FiQCISampler

   # Initialize sampler with mitigation level 1
   sampler = FiQCISampler(backend, mitigation_level=1, calibration_shots=2000, calibration_file="cals.json")

For more details see the API reference documentation for :class:`~fiqci.ems.FiQCISampler`.

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
     - Dynamical Decoupling standard sequence (see :ref:`below <fiqci-sampler-dd>`)
   * - 3
     - Level 2 + Pauli Twirling
     - Pauli Twirling with 10 twirls on all two-qubit gates

Mitigation Options
------------------

Mitigators can also be configured manually using the provided methods.

- :ref:`Readout Error Mitigation (REM) <fiqci-sampler-rem>`
- :ref:`Dynamical Decoupling (DD) <fiqci-sampler-dd>`
- :ref:`Pauli Twirling <fiqci-sampler-pt>`

.. _fiqci-sampler-rem:

REM (Readout Error Mitigation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Readout error mitigation uses M3 (matrix-free measurement mitigation) to correct measurement errors. It is enabled by default at mitigation level 1.

Configure REM using the :meth:`~fiqci.ems.FiQCISampler.rem` method:

.. code-block:: python

   sampler.rem(enabled=True, calibration_shots=2000, calibration_file="cals.json")

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

.. _fiqci-sampler-dd:

Dynamical Decoupling (DD)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamical decoupling inserts sequences of gates to mitigate decoherence. It is enabled at mitigation level 2.

Configure DD using the :meth:`~fiqci.ems.FiQCISampler.dd` method:

.. code-block:: python

   sampler.dd(enabled=True, gate_sequences=None) # None uses a standard set of sequences

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
             - ``"alap"``: Apply the sequence *as late as possible* whenever the idle period exceeds the threshold.
             - ``"center"``: Apply the sequence centered within idle periods exceeding the threshold.

.. _fiqci-sampler-pt:

Pauli Twirling
~~~~~~~~~~~~~~~~

Pauli twirling sandwiches two-qubit gates with random single-qubit Pauli gates to mitigate coherent errors. It is enabled at mitigation level 3.

Configure Pauli Twirling using the :meth:`~fiqci.ems.FiQCISampler.pauli_twirl` method:

.. code-block:: python

   sampler.pauli_twirl(enabled=True, num_twirls=10, gates_to_twirl=None) # None twirls all two-qubit gates

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

Use the :attr:`~fiqci.ems.FiQCISampler.mitigator_options` property to view currently applied mitigation settings:

.. code-block:: python

   sampler.mitigator_options

Counting Circuits
-----------------

Use :meth:`~fiqci.ems.FiQCISampler.total_circuits_generated` to see how many circuits will actually be executed under the current mitigation settings (accounts for Pauli twirls):

.. code-block:: python

   sampler.total_circuits_generated(num_base_circuits=len(circuits), detailed=True)

Set ``detailed=True`` to print the breakdown and return a dictionary with each multiplier; otherwise the method returns the total as an ``int``. The count does not include REM calibration circuits.

Running Circuits
----------------

:meth:`~fiqci.ems.FiQCISampler.run` accepts a ``max_batch_size`` parameter (default ``100``) that controls how many circuits are sent to the backend in a single job. Larger circuit lists are split into batches automatically; results are re-combined so indexing matches the order of the input circuits.

.. code-block:: python

   job = sampler.run(circuits, shots=1024, max_batch_size=100)

``run`` returns **immediately** with a lazy job handle without waiting for results: a :class:`~fiqci.ems.backend.jobs.BatchedJob` at level 0, or a :class:`~fiqci.ems.backend.jobs.MitigatedJob` at level 1+. The per-batch ``job_id()`` values and an aggregated ``status()`` are available right away; error mitigation and result combination are computed the first time you call ``job.result()`` (which then exposes a single combined ``Result`` indexed in input order, and caches it).

The handle also lets you inspect a multi-batch run before it finishes:

.. code-block:: python

   job.job_ids()         # backend job id of every batch (None for any unsubmitted batch)
   job.statuses()        # per-batch JobStatus, in submission order
   job.status()          # single aggregated status across all batches
   job.done()            # True once every batch has reached a terminal state
   job.partial_results() # per-batch results for batches that have already completed

If a batch fails, ``job.result()`` raises :class:`~fiqci.ems.BatchFailedError` identifying the failing batch and the input circuit indices it covered, instead of an opaque error during combination.

.. note::

   Submission is not atomic. If the backend rejects a circuit partway through (e.g. an un-transpiled circuit), ``run`` does **not** raise: it logs a warning, stops submitting further batches, and still returns a handle covering every intended batch. Submitted batches keep their job ids and status, the rejected batch reports ``ERROR``, and the batches skipped afterwards report ``CANCELLED`` (with ``job_id()`` ``None``). Inspect the outcome with ``statuses()``/``status()``/``partial_results()``; ``result()`` then raises :class:`~fiqci.ems.BatchFailedError` because a complete combined result cannot be formed.


Examples
--------

- :doc:`Using The FiQCI Sampler <../notebooks/sampling_fiqci_sampler>`
- :doc:`Advanced Readout Error Mitigation <../notebooks/advanced_readout_error_mitigation_m3>`
