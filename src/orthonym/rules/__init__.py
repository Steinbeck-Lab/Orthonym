"""
Rules module - IUPAC nomenclature rules implementation.

Contains the classification logic that applies IUPAC rules to
determine naming strategy, principal groups, and locants.
"""

from .seniority import get_principal_group, SENIORITY_ORDER, get_suffix, get_prefix

__all__ = [
    "get_principal_group",
    "SENIORITY_ORDER",
    "get_suffix",
    "get_prefix",
]
