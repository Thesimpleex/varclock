"""varclock: the market's variance calendar from option-chain snapshots.

Not affiliated with, endorsed by or sponsored by Cboe Global Markets or S&P Dow Jones Indices.
"""

from __future__ import annotations

from .analysis import CalendarResult, build_calendar
from .clock import ClockBootstrap, ClockData, ClockFit, bootstrap_clock, fit_clock
from .events import Event, load_events
from .indices import INDEX_SPECS, IndexResult, compute_index, replicate
from .quotes import OptionStrip, Snapshot
from .rates import YieldCurve
from .strip import ForwardVarianceStrip, build_strip
from .variance import TermVariance, VarianceError, interpolate_variance, single_term_variance

__version__ = "0.1.0"

__all__ = [
    "INDEX_SPECS",
    "CalendarResult",
    "ClockBootstrap",
    "ClockData",
    "ClockFit",
    "Event",
    "ForwardVarianceStrip",
    "IndexResult",
    "OptionStrip",
    "Snapshot",
    "TermVariance",
    "VarianceError",
    "YieldCurve",
    "__version__",
    "bootstrap_clock",
    "build_calendar",
    "build_strip",
    "compute_index",
    "fit_clock",
    "interpolate_variance",
    "load_events",
    "replicate",
    "single_term_variance",
]
