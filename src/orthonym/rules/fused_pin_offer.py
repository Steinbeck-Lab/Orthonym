"""The fusion name of a molecule when the older fused-heterocycle namer's name is known not
to be the PIN and the bridged fused PIN builder names the same molecule better.

``fused_rings.name_fused_heterocycle`` is asked first for every fused ring system. It builds
fusion descriptors from the two components of the system and, for a skeleton that has a
retained name, spells the descriptor instead: 'benzo[d]pyrimidine' for quinazoline,
'pyrazino[2,3-d]pyrimidine' for pteridine, '4H-cyclohexa[b]pyran' for 4H-1-benzopyran.
 "Retained names for heterocycles (also called trivial names) with the maximum
number of noncumulative double bonds (called mancude ring systems) used for parent
components and as attached components are given in Table 2.8" (the Blue Book), and
Table 2.8 gives '(9) quinazoline (PIN)' (:11553). The namer records such a spelling as a
non-PIN fragment (``fused_rings._record_non_pin_spelling``), as it does a hydro name that
puts a ketone on a saturated position of a hydro core without the added hydrogen
,:24766;,:24689), so the name it ships is labelled below the PIN and the default tier
declines the molecule. The builder (``bridged_fused_pin.build_fused``) names the same
molecules from the PIN parents ('1,4-dihydroquinazolin-2-amine', SUBSTITUENT GROUPS
MODIFIED BY THE PREFIXES 'HYDRO' OR 'DEHYDRO',:16870) but is only tried when the older
namer declines (``bridged_fused.name_fused_carbocycle_pin``), so for these molecules it
never ran.

This module offers the builder's name beside the older namer's, as two candidates for one
molecule (the contributor guide a project rule). The older name stays unless ALL of these hold:

* the older name carries a recorded non-PIN part (``known_non_pin``: the name-scoped records
  and the non-PIN vocabulary, the test that labels a shipped name below the PIN);
* the builder names the molecule, and its name carries no recorded non-PIN part (an offer
  that is labelled below the PIN as well gains no tier);
* OPSIN reads the builder's name back to the molecule's full standard InChI (constitution,
  stereo, charge, isotopes), the check that gives a name the 'verified' of its tier, AND to the
  same canonical SMILES (the InChI does not see a bond-shift partner, see ``_reads_back``);
* the offered naming covers the whole ring system (the check the caller applies to the
  older name).

The label of the name that ships is still decided downstream (the namer's own validity gate
and the independent spelling check), so a builder name that fails the spelling check ships
as ``systematic_verified``, the label the older name had. A molecule the builder declines,
or without a JVM, keeps the older name.
"""
from typing import Callable, Optional


def known_non_pin(prov: dict, name: str) -> bool:
    """True when the label logic would lower ``name`` below the PIN for a reason that is
    NAME-scoped: a recorded non-PIN fragment or label it contains, or non-PIN vocabulary.
    The whole-call flags (a breadth producer's ring prefix, the PIN tier's promotion
    re-run) and the uncertified PIN names are left out, as in the namer's own
    ``_known_non_pin``: they lower every name of the call equally, so they cannot
    separate the two candidates."""
    from ..metrics.provenance import name_carries_non_pin_part
    return name_carries_non_pin_part(
        dict(prov, pin_promotion_rerun=False, general_ring_prefix=False,
             uncertified_pin_names=()), name)


def _reads_back(mol, name: str) -> bool:
    """True iff OPSIN reads ``name`` back to the molecule as DRAWN: the same standard InChI
    (the check that gives a name its 'verified') and the same canonical SMILES. The InChI
    alone does not see where the localized double bonds of a non-aromatic mancude ring
    system sit, so the name of a bond-shift partner passes it: "Localized double
    bonds" (the Blue Book) "If it is necessary to identify isomers that differ only by
    virtue of the location of localized double bonds, this differentiation is indicated by the
    use of the Greek letter Δ" ('1,6-dimethyl-Δ1-heptalene (PIN)'), so '1-methylheptalene'
    is not a name of the drawing whose own name is '5-methylheptalene'. False when OPSIN
    cannot be run or the name does not parse (fail closed: the older name stays)."""
    try:
        from rdkit import Chem

        from ..validation.opsin_roundtrip import opsin_roundtrip_check
        from ..validation.twin_spellings import opsin_spelling
        smiles = Chem.MolToSmiles(mol)
        if not smiles:
            return False
        res = opsin_roundtrip_check(smiles, opsin_spelling(name))
        if res.get("passed") is not True:
            return False
        back = Chem.MolFromSmiles(res.get("opsin_smiles") or "")
        return back is not None and Chem.MolToSmiles(back) == smiles
    except Exception:  # noqa: BLE001 - a failed probe must keep the older name
        return False


def name_fused_with_pin_offer(mol, covers_ring_system: Callable) -> Optional[tuple]:
    """``fused_rings.name_fused_heterocycle(mol)``, or the bridged fused PIN builder's
    tuple for the same molecule when it is a verified, PIN-form replacement of a name that
    carries a recorded non-PIN part (see the module doc). Both are the complex-ring tuple
    ``(name, ring_atoms, atom_to_locant, substituents_included)``, or None.

    ``covers_ring_system(mol, atom_to_locant)`` is the caller's coverage check of a fused
    core (``composer._fused_core_covers_ring_system``), applied to the offered tuple."""
    from .fused_rings import name_fused_heterocycle
    older = name_fused_heterocycle(mol)
    if older is None:
        return None
    from ..metrics.provenance import get_provenance
    if not known_non_pin(get_provenance(), older[0]):
        return older
    from .bridged_fused import name_fused_carbocycle_pin
    offered = name_fused_carbocycle_pin(mol)
    if offered is None or offered[0] == older[0]:
        return older
    if known_non_pin(get_provenance(), offered[0]):
        return older
    if not covers_ring_system(mol, offered[2]) or not _reads_back(mol, offered[0]):
        return older
    return offered
