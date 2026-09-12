"""Per-naming-call cache of a molecule's atom and bond tuples (audit 2026-09-03, S2).

Why: ``for a in mol.GetAtoms`` runs through RDKit's Python sequence wrapper
(``rdkit/Chem/__init__.py``: ``__iter__`` -> ``__getitem__`` -> ``_sizeCalc`` per
step). A 58-atom walk costs about 88 us that way, 53 us through
``GetAtomWithIdx`` and 14 us over a materialised tuple. The engine walks the
same input molecule hundreds of times per name -- every dispatch predicate
scans the atoms once -- so on a dev split this wrapper was 24% of engine CPU.

What ``atoms_of(mol)`` / ``bonds_of(mol)`` promise:

* **Same atoms, same order** as ``mol.GetAtoms`` / ``mol.GetBonds``. The
  tuple is built from ``GetAtomWithIdx(i)`` for ``i`` in index order, which is
  exactly the wrapper's iteration order.
* **Scope = one top-level naming call.** The tuple lives in the memo scope that
  ``assembly.memo`` opens around ``name``; with no open scope nothing is cached
  and every call rebuilds. Cross-molecule staleness is structurally impossible.
* **Never a stale tuple.** Only immutable-by-convention ``Chem.Mol`` objects are
  cached; a ``Chem.RWMol`` (the only type on which atoms can be removed,
  replaced or added) is always rebuilt. The entry keeps a strong reference to
  the molecule, so ``id(mol)`` cannot be recycled by a new molecule while the
  entry exists, and a hit is also checked against the live atom/bond count.
* **Verify mode.** ``ORTHONYM_MOLCACHE=verify`` rebuilds on every hit and raises
  :class:`MolCacheMismatchError` if any cached atom's identity, element or charge
  differs from the live molecule -- the same continuous completeness check
  ``assembly.memo`` offers. ``ORTHONYM_MOLCACHE=off`` disables caching.

Atom and bond wrappers reference the live C++ objects, so property edits made
in place on a ``Chem.Mol`` (aromaticity, charges, isotopes) are visible through
the cached tuple exactly as through a fresh ``GetAtoms``.
"""
from __future__ import annotations

import os
from typing import Tuple

import functools

from rdkit import Chem
from rdkit.Chem import inchi as _inchi

from ..assembly.memo import _cache_var

_MODE = os.environ.get("ORTHONYM_MOLCACHE", "on").strip().lower()
if _MODE not in ("on", "off", "verify"):
    _MODE = "on"

_NS_ATOMS = "molcache.atoms"
_NS_BONDS = "molcache.bonds"
_NS_INCHIKEY = "molcache.inchikey"


class MolCacheMismatchError(AssertionError):
    """Verify mode: a cached atom/bond tuple no longer matches the live molecule."""


def _fresh_atoms(mol) -> Tuple[Chem.Atom, ...]:
    get = mol.GetAtomWithIdx
    return tuple(get(i) for i in range(mol.GetNumAtoms()))


def _fresh_bonds(mol) -> Tuple[Chem.Bond, ...]:
    get = mol.GetBondWithIdx
    return tuple(get(i) for i in range(mol.GetNumBonds()))


def _atom_sig(atoms):
    return [(a.GetIdx(), a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope()) for a in atoms]


def _bond_sig(bonds):
    return [(b.GetIdx(), b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondType())) for b in bonds]


def _cached(mol, ns, fresh, count, sig):
    if _MODE == "off" or isinstance(mol, Chem.RWMol):
        return fresh(mol)
    cache = _cache_var.get()
    if cache is None:
        return fresh(mol)
    key = (ns, id(mol))
    entry = cache.get(key)
    if entry is not None and entry[0] is mol and len(entry[1]) == count(mol):
        if _MODE == "verify":
            live = fresh(mol)
            if sig(live) != sig(entry[1]):
                raise MolCacheMismatchError(ns, id(mol))
        return entry[1]
    value = fresh(mol)
    cache[key] = (mol, value)   # the strong ref to mol pins id(mol) for the scope
    return value


def atoms_of(mol) -> Tuple[Chem.Atom, ...]:
    """``tuple(mol.GetAtoms)`` in index order, cached per naming call."""
    return _cached(mol, _NS_ATOMS, _fresh_atoms, Chem.Mol.GetNumAtoms, _atom_sig)


def bonds_of(mol) -> Tuple[Chem.Bond, ...]:
    """``tuple(mol.GetBonds)`` in index order, cached per naming call."""
    return _cached(mol, _NS_BONDS, _fresh_bonds, Chem.Mol.GetNumBonds, _bond_sig)


def _inchi_sig(mol):
    """Everything an in-place edit could change that the standard InChI reads: element, charge,
    isotope, explicit H count and chiral tag per atom; bond order and bond stereo per bond.
    Cheap (tens of microseconds via the cached tuples) against ~1 ms for the InChIKey."""
    return (tuple((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(), a.GetNumExplicitHs(),
                   int(a.GetChiralTag())) for a in atoms_of(mol)),
            tuple((int(b.GetBondType()), int(b.GetStereo())) for b in bonds_of(mol)))


def inchikey_of(mol) -> str:
    """``Chem.MolToInchiKey(mol)`` cached per mol object within the naming scope (Lever F,
    2026-09-12). The entry pins ``mol`` (so its id cannot be recycled) and stores the
    :func:`_inchi_sig` fingerprint; a hit is served only when the live fingerprint is equal,
    so an in-place edit of charge, isotope, H count, chirality or bond order/stereo forces a
    recompute. RWMol is never cached. Verify mode recomputes on every hit and raises on a
    difference."""
    if _MODE == "off" or isinstance(mol, Chem.RWMol):
        return _inchi.MolToInchiKey(mol)
    cache = _cache_var.get()
    if cache is None:
        return _inchi.MolToInchiKey(mol)
    key = (_NS_INCHIKEY, id(mol))
    sig = _inchi_sig(mol)
    entry = cache.get(key)
    if entry is not None and entry[0] is mol and entry[2] == sig:
        if _MODE == "verify" and _inchi.MolToInchiKey(mol) != entry[1]:
            raise MolCacheMismatchError(_NS_INCHIKEY, id(mol))
        return entry[1]
    value = _inchi.MolToInchiKey(mol)
    cache[key] = (mol, value, sig)
    return value


@functools.lru_cache(maxsize=65536)
def canon_smiles(smi: str) -> str:
    """``Chem.CanonSmiles(smi)``: a pure function of the string, so a process-wide LRU is exact."""
    return Chem.CanonSmiles(smi)


def cached_by_key(mol, ns: str, key, fresh):
    """Memoise ``fresh`` under ``(ns, id(mol), key)`` for the naming scope (Lever E, 2026-09-12).
    The entry pins ``mol`` so its id cannot be recycled; RWMol is never cached; verify mode
    recomputes on every hit and raises:class:`MolCacheMismatchError` on a difference. The caller
    puts everything the value depends on into ``key`` (see ring_selection.ring_system_score)."""
    if _MODE == "off" or isinstance(mol, Chem.RWMol):
        return fresh()
    cache = _cache_var.get()
    if cache is None:
        return fresh()
    ck = (ns, id(mol), key)
    entry = cache.get(ck)
    if entry is not None and entry[0] is mol:
        if _MODE == "verify":
            live = fresh()
            if live != entry[1]:
                raise MolCacheMismatchError(ns, id(mol), key)
        return entry[1]
    value = fresh()
    cache[ck] = (mol, value)
    return value
