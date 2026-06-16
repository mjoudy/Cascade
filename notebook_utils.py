"""
Compatibility shim for `from notebook_utils import *`.

The helpers that used to live here have moved into the `functions` package:

    functions/groups.py    — dataset grouping schemes (GROUPS, DS_TO_GROUP, …)
    functions/plotting.py  — all plot_* helpers and correlation-hist constants

This module now just re-exports the full public API the notebooks expect, so
existing `from notebook_utils import *` / `from notebook_utils import GROUPS`
imports keep working. New code should import from `functions.*` directly.
"""

from functions.data_manager import *
from functions.derivative_method import *
from functions.metrics import *
from functions.reconstruction import *
from functions.groups import *
from functions.plotting import *
