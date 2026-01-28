"""
Assembly module - Name fragment ordering and string construction.

This module combines name fragments into complete IUPAC names:
- Prefix generation (substituents, ordered alphabetically)
- Suffix generation (principal group)
- Parent name construction
- Stereodescriptor formatting
"""

from .composer import assemble_name

__all__ = ["assemble_name"]
