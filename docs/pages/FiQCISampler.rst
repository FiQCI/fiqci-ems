FiQCISampler
============

``FiQCISampler`` is a sampling interface that wraps an IQM backend and applies error mitigation to measurement results. It executes quantum circuits and returns mitigated counts.

For general usage instructions, see :ref:`Usage <usage-sampler>`.

Mitigation Options
------------------

REM (Readout Error Mitigation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Readout error mitigation uses M3 (matrix-free measurement mitigation) to correct measurement errors. It is enabled by default at mitigation level 1.

Configure REM using the ``rem()`` method:

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
     - Level 2 + additional
     - TBD

Inspecting Options
------------------

Use the ``mitigator_options`` property to view currently applied mitigation settings:

.. code-block:: python

   sampler.mitigator_options

Examples
--------

- :doc:`Using The FiQCI Sampler <../notebooks/sampling_fiqci_sampler>`
- :doc:`Advanced Readout Error Mitigation <../notebooks/advanced_readout_error_mitigation_m3>`
