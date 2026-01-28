"""
Data module - Lookup tables and reference data.

Contains:
- Retained (trivial) names that take precedence over systematic names
- Ring system names and numbering
- Chain prefixes
- Functional group mappings
"""

from .retained_names import RETAINED_NAMES, get_retained_name

__all__ = ["RETAINED_NAMES", "get_retained_name"]
