"""FiQCI Error Mitigation Service.

This package provides error mitigation capabilities for quantum computations
running on FiQCI quantum computers via HPC systems.
"""

from fiqci.ems.rem import M3IQM, apply_readout_error_mitigation, readout_error_m3

__all__ = ["M3IQM", "apply_readout_error_mitigation", "readout_error_m3"]
