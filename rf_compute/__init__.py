"""rf-compute: wave-domain computation kernel — package init."""

from .rf_compute import (
    Operator, AirCompOperator, LatticeAirCompOperator, FadingAirCompOperator,
    ConvolutionOperator, InversionOperator, ReservoirOperator,
    WaveComputeKernel,
    boxcar, differencer, matched, hilbert,
)
from . import lattice
from . import coefficients
from .lattice import (
    mod_lattice, encode, decode, channel, run_trial, monte_carlo,
    fading_trial, fading_scoreline,
)
from .coefficients import (
    mmse_alpha, computation_rate, norm_bound, select_coefficients,
    fading_gains, lll_reduce,
)

__version__ = "0.1.0"
__all__ = [
    'Operator', 'AirCompOperator', 'LatticeAirCompOperator',
    'FadingAirCompOperator',
    'ConvolutionOperator', 'InversionOperator', 'ReservoirOperator',
    'WaveComputeKernel',
    'boxcar', 'differencer', 'matched', 'hilbert',
    'lattice', 'coefficients',
    'mod_lattice', 'encode', 'decode', 'channel', 'run_trial', 'monte_carlo',
    'fading_trial', 'fading_scoreline',
    'mmse_alpha', 'computation_rate', 'norm_bound', 'select_coefficients',
    'fading_gains', 'lll_reduce',
    '__version__',
]
