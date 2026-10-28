"""strands-decider: a System One model -- typed, calibrated answers instead of text."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("strands-decider")
except PackageNotFoundError:
    # Editable install without installed metadata (e.g. a fresh checkout
    # that was never `pip install -e .`'d). Fall back to a sentinel so
    # imports still succeed.
    __version__ = "0.0.0+unknown"
