"""Orthonym: IUPAC names for chemical structures.

Orthonym reads a structure written as SMILES and builds its IUPAC name from
the rules of the IUPAC 2013 recommendations, aiming at the Preferred IUPAC
Name. Before a name is returned, OPSIN reads it back into a structure and that
structure is compared with yours. When no name passes, you get a label that
says so instead of a name.

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

__version__ = "1.0.0"  # x-release-please-version
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
