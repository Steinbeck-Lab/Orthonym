"""Orthonym: IUPAC names for chemical structures.

Orthonym reads a structure written as SMILES and builds its IUPAC name from
the rules of the IUPAC 2013 recommendations, aiming at the Preferred IUPAC
Name. Before a name is returned, OPSIN reads it back into a structure and that
structure is compared with yours. The default tier returns a name only when
the strict path for the Preferred IUPAC Name built it and verified it; its only
exceptions are a few names OPSIN cannot read (retained natural-product and
metal-complex names from exact-match lists, a few name forms outside OPSIN's
grammar, and stereodescriptors OPSIN cannot parse), returned without that full
read-back and marked in the provenance row (``Orthonym.name_tiered``). When
no name passes, you get a label that says so instead of a name; the wider tiers
also return names that are not the preferred name.

The public names are ``name_compound`` (one molecule, one name),
``name_with_tree`` (the name and its parts), the ``Orthonym`` class (all
options, and the provenance row), ``NamingResult``, ``NameTreeNode``,
``classify_limit`` and ``OrthonymLimitError``.

Examples
--------
>>> from orthonym import name_compound
>>> name_compound("CCO")
'ethanol'
>>> name_compound("CC(=O)O")
'acetic acid'
"""

__version__ = "1.0.3"  # x-release-please-version
__author__ = "Kohulan Rajan"

from .assembly.name_tree import NameTreeNode, NamingResult
from .errors import OrthonymLimitError
from .namer import Orthonym, classify_limit, name_compound, name_with_tree

__all__ = [
    "name_compound",
    "name_with_tree",
    "Orthonym",
    "NameTreeNode",
    "NamingResult",
    "OrthonymLimitError",
    "classify_limit",
    "__version__",
]
