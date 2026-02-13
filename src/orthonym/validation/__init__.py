"""
Orthonym Validation Module

Tools for validating generated IUPAC names against external resolvers
(OPSIN, PubChem) and measuring name quality (atom coverage).
"""

from .pubchem_validator import load_cache, lookup_name_pubchem, save_cache
from .dual_validator import DualResult, validate_compound

__all__ = [
    "lookup_name_pubchem",
    "load_cache",
    "save_cache",
    "DualResult",
    "validate_compound",
]

# Conditionally import atom_coverage if it exists (added by plan 45-01)
try:
    from .atom_coverage import CoverageResult, validate_atom_coverage

    __all__.extend(["CoverageResult", "validate_atom_coverage"])
except ImportError:
    pass
