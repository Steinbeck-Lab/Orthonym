"""The systematic name of the caller's whole molecule, offered before a Chapter name.

 INTRODUCTION (the Blue Book): "When the full structure is known, a 'systematic name' may be generated in
accordance with Rules described in Chapters through... Preferred IUPAC names (PINs) are not identified for the
compounds in this Chapter." A producer of a Chapter name (a retained amino-acid name, a natural-product name)
asks:func:`systematic_pin_offer` first: the molecule is named once more by a default-tier engine with that producer's
names left out (its context variable set), under the default tier's context (``assembly.default_tier_scope``: a
best-effort caller's breadth is off, so both tiers are offered the same name). The name is returned only when it reads
as a PIN (the checks in:func:`_reads_as_pin`) and OPSIN reads it back to the full InChIKey of the molecule. Else None,
and the producer keeps its own name, labelled below the PIN by the spelling check
(``validation/spelling/checks_natural_products.py``).
"""
from __future__ import annotations

import contextvars
from typing import Optional

from rdkit import Chem


def systematic_pin_offer(mol, off: contextvars.ContextVar, key: str, *, stereo_unread_ok: bool = False) -> Optional[str]:
    """The systematic PIN of ``mol`` named with ``off`` set, or None: only for the caller's whole molecule
    (``outermost_name_input``; a peptide residue or a salt component keeps the name its producer asks for), not while
    ``off`` is set, memoised per molecule under ``key``. ``stereo_unread_ok``: also a name the nested naming labels
    pin_unverified because OPSIN reads its constitution but not its stereodescriptors ('(1R,3r,5S)-8-methyl-8-
    azabicyclo[3.2.1]octan-3-yl...'), the name the PIN tier ships for such a molecule; the producer's own name had
    that licence as a PIN, and has none since it is not one."""
    if mol is None or off.get():
        return None
    from ..assembly.fragment_naming import outermost_name_input
    try:
        smiles = Chem.MolToSmiles(mol)
        outer = outermost_name_input()
        outer_mol = Chem.MolFromSmiles(outer) if outer else None
        if outer_mol is None or Chem.MolToSmiles(outer_mol) != smiles:
            return None
    except Exception:
        return None
    from ..assembly.nested_memo import cached_nested_call
    return cached_nested_call(key, (smiles,), lambda: _named_with(mol, smiles, off, stereo_unread_ok))


def _named_with(mol, smiles: str, off: contextvars.ContextVar, stereo_unread_ok: bool) -> Optional[str]:
    from ..assembly.default_tier_scope import default_tier_scope
    from ..metrics import provenance as _pv
    snap = _pv.get_provenance()             # name_tiered clears the provenance; the caller's records are put back
    token = off.set(True)
    try:
        from ..errors import is_failure_name
        from ..namer import Orthonym
        with default_tier_scope():
            row = Orthonym(style="pin").name_tiered(smiles)
        name = row.get("name")
        if not name or is_failure_name(name):
            return None
        stereo_unread = (stereo_unread_ok and row.get("tier") == "pin_unverified"
                         and row.get("gate_outcome") == "self_consistency_constitution_only")
        # read against the nested naming's own records, before the caller's are put back
        return name if _reads_as_pin(mol, name, stereo_unread) else None
    except Exception:                       # the offer must never break a naming: the producer's name stands
        return None
    finally:
        off.reset(token)
        _pv.restore_provenance(snap)


def _reads_as_pin(mol, name: str, stereo_unread: bool = False) -> bool:
    """A nested name call does not apply the default tier's emission rule, so the PIN status is read here: no part
    the naming recorded as not the PIN, no broken spelling rule (the check among them), no molecule whose PIN is
    multiplicative; 'L-cystine' keeps its name rather than a substitutive one), and a full-InChIKey read-back."""
    from ..metrics.provenance import get_provenance, name_carries_non_pin_part
    from ..validation.coverage_gate import _full_inchikey_roundtrips
    from ..validation.pin_spelling import check_pin_spelling
    from .pin_vocabulary import multiplicative_pin_expected
    return not (name_carries_non_pin_part(get_provenance(), name) or check_pin_spelling(mol, name)
                or multiplicative_pin_expected(mol) or not (stereo_unread or _full_inchikey_roundtrips(mol, name)))
