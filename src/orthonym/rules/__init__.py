"""
Rules module - IUPAC nomenclature rules implementation.

Contains the classification logic that applies IUPAC rules to
determine naming strategy, principal groups, and locants.
"""

from .seniority import SENIORITY_ORDER, get_prefix, get_principal_group, get_suffix

__all__ = [
    "get_principal_group",
    "SENIORITY_ORDER",
    "get_suffix",
    "get_prefix",
]
