"""
Orthonym Validation Module

Tools for validating generated IUPAC names against external resolvers
(OPSIN, PubChem) and measuring name quality (atom coverage).
"""

from .dual_validator import DualResult, validate_compound
from .format_validator import validate_name_format
from .opsin_grammar import (
    OpsinGrammar,
    opsin_grammar_suggest_fix,
    opsin_grammar_validate,
)
from .opsin_roundtrip import opsin_parse, opsin_roundtrip_check
from .pubchem_validator import load_cache, lookup_name_pubchem, save_cache

__all__ = [
    "lookup_name_pubchem",
    "load_cache",
    "save_cache",
    "DualResult",
    "validate_compound",
    "validate_name_format",
    "opsin_parse",
    "opsin_roundtrip_check",
    "OpsinGrammar",
    "opsin_grammar_validate",
    "opsin_grammar_suggest_fix",
]

# Conditionally import atom_coverage if it exists (added by plan 45-01)
try:
    from .atom_coverage import CoverageResult, validate_atom_coverage

    __all__.extend(["CoverageResult", "validate_atom_coverage"])
except ImportError:
    pass
