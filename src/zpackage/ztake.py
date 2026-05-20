"""Public Ztake API.

This module re-exports the refactored Ztake implementation so users can import
``zpackage.ztake`` while the root-level files remain untouched.
"""

from .ztake_refactored import *  # noqa: F401,F403
