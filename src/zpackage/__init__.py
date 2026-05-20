"""zpackage: utilities for CMIP6 ocean and climate analysis."""

try:
    from .ztake_refactored import Ztake
except Exception:  # pragma: no cover - optional runtime dependencies may be unavailable
    Ztake = None

__all__ = ["Ztake"]
