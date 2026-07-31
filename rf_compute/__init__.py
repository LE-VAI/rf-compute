"""rf-compute: wave-domain computation kernel — package init."""

from .rf_compute import (
    Operator, AirCompOperator, ConvolutionOperator, InversionOperator, ReservoirOperator,
    WaveComputeKernel,
    boxcar, differencer, matched, hilbert,
)

__version__ = "0.1.0"
__all__ = [
    'Operator', 'AirCompOperator', 'ConvolutionOperator', 'InversionOperator', 'ReservoirOperator',
    'WaveComputeKernel',
    'boxcar', 'differencer', 'matched', 'hilbert',
    '__version__',
]