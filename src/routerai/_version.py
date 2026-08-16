"""Single source of the package version.

Kept in its own leaf module so ``_http`` can read it for the User-Agent
without importing the package root, which would be a cycle.
"""

__version__ = "0.3.0rc1"
