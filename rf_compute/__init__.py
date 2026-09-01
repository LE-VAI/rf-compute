"""rf-compute: wave-domain computation kernel — package init."""

from .rf_compute import (
    Operator, AirCompOperator, LatticeAirCompOperator,
    ConvolutionOperator, InversionOperator, ReservoirOperator,
    WaveComputeKernel,
    boxcar, differencer, matched, hilbert,
)
from . import lattice
from .lattice import mod_lattice, encode, decode, channel, run_trial, monte_carlo

__version__ = "0.1.0"
__all__ = [
    'Operator', 'AirCompOperator', 'LatticeAirCompOperator',
    'ConvolutionOperator', 'InversionOperator', 'ReservoirOperator',
    'WaveComputeKernel',
    'boxcar', 'differencer', 'matched', 'hilbert',
    'lattice', 'mod_lattice', 'encode', 'decode', 'channel', 'run_trial', 'monte_carlo',
    '__version__',
]