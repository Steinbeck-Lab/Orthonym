"""
Orthonym - Open Structure-To-IUPAC-Name Generator

A rule-based system to generate IUPAC names from molecular structures (SMILES).
First comprehensive open-source implementation targeting IUPAC 2013 (Blue Book) compliance.

Example usage:
    >>> from orthonym import name_compound
    >>> name_compound("CCO")
    'ethanol'
    >>> name_compound("CC(=O)O")
    'acetic acid'
"""

__version__ = "0.1.0"
__author__ = "Kohulan Rajan"

from .namer import name_compound, Orthonym, name_with_tree
from .assembly.name_tree import NameTreeNode, NamingResult

__all__ = [
    "name_compound",
    "name_with_tree",
    "Orthonym",
    "NameTreeNode",
    "NamingResult",
    "__version__",
]
