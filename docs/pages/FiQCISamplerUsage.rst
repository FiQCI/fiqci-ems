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
     - Level 1 + additional
     - TBD
   * - 3
     - Level 2 + additional
     - TBD

Mitigation Options
------------------

Mitigators can also be configured manually using the provided methods.

- :ref:`Readout Error Mitigation (REM) <fiqci-sampler-rem>`
- :ref:`Dynamical Decoupling (DD) <fiqci-sampler-dd>`

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
     - List of (treshold_length, sequence, strategy) tuples defining DD behavior. If ``None``, uses a standard set of sequences.
         - ``treshold_length``: Minimum circuit length to apply the sequence. If ``None``, uses ``len(sequence)`` or 2 if sequence is ``None``.
         - ``sequence``: List of gate names or :class:`~fiqci.ems.primitives.prx_sequence.PRXSequence` defining the DD sequence.
         - ``strategy``: Strategy for applying the sequence. One of:
             - ``"asap"``: Apply the sequence as soon as possible whenever the idle period exceeds the threshold.
             - ``"alap"``: Apply the sequence as late as possible whenever the idle period exceeds the threshold.
             - ``"center"``: Apply the sequence centered within idle periods exceeding the threshold.


Inspecting Options
------------------

Use the :attr:`~fiqci.ems.FiQCISampler.mitigator_options` property to view currently applied mitigation settings:

.. code-block:: python

   sampler.mitigator_options

Examples
--------

- :doc:`Using The FiQCI Sampler <../notebooks/sampling_fiqci_sampler>`
- :doc:`Advanced Readout Error Mitigation <../notebooks/advanced_readout_error_mitigation_m3>`
