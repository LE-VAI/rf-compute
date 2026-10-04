"""rf-compute: wave-domain computation kernel — package init."""

from .rf_compute import (
    Operator, AirCompOperator, LatticeAirCompOperator, FadingAirCompOperator,
    OTAAggregationOperator, ConvolutionOperator, InversionOperator, ReservoirOperator,
    WaveComputeKernel,
    boxcar, differencer, matched, hilbert,
)
from . import lattice
from . import coefficients
from . import ota_fl
from . import hardware
from .lattice import (
    mod_lattice, encode, decode, channel, run_trial, monte_carlo,
    fading_trial, fading_scoreline,
)
from .coefficients import (
    mmse_alpha, computation_rate, norm_bound, select_coefficients,
    fading_gains, lll_reduce,
)
from .ota_fl import (
    make_federated_data, gradient_spread, ota_aggregate, aggregation_quality,
)
from .hardware import (
    HardwareImpairments, PHONE_GRADE, DEV_BOARD_UNCALIBRATED, DEV_BOARD_CLOCKED,
    esp_iq_word_layout, pack_esp_iq, unpack_esp_iq, ota_aggregate_hardware,
    survival_table, overhead_table,
)

__version__ = "0.1.2"
__all__ = [
    'Operator', 'AirCompOperator', 'LatticeAirCompOperator',
    'FadingAirCompOperator', 'OTAAggregationOperator',
    'ConvolutionOperator', 'InversionOperator', 'ReservoirOperator',
    'WaveComputeKernel',
    'boxcar', 'differencer', 'matched', 'hilbert',
    'lattice', 'coefficients', 'ota_fl', 'hardware',
    'mod_lattice', 'encode', 'decode', 'channel', 'run_trial', 'monte_carlo',
    'fading_trial', 'fading_scoreline',
    'mmse_alpha', 'computation_rate', 'norm_bound', 'select_coefficients',
    'fading_gains', 'lll_reduce',
    'make_federated_data', 'gradient_spread', 'ota_aggregate', 'aggregation_quality',
    'HardwareImpairments', 'PHONE_GRADE', 'DEV_BOARD_UNCALIBRATED', 'DEV_BOARD_CLOCKED',
    'esp_iq_word_layout', 'pack_esp_iq', 'unpack_esp_iq', 'ota_aggregate_hardware',
    'survival_table', 'overhead_table',
    '__version__',
]
