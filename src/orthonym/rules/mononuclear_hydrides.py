"""Mononuclear-hydride halides bearing the λ-convention (P-68 / P-31.1.4.2).

Names a single non-carbon "hub" atom (a chalcogen S/Se/Te, phosphorus, or
iodine) whose substituents are ALL halogens and whose valence is *non-standard*
(so the λ-convention applies), e.g.::

    FS(F)(F)(F)(F)F   -> hexafluoro-lambda6-sulfane   (SF6)
    FS(F)(F)F         -> tetrafluoro-lambda4-sulfane   (SF4)
    FP(F)(F)(F)F      -> pentafluoro-lambda5-phosphane (PF5)
    ClP(Cl)(Cl)(Cl)Cl -> pentachloro-lambda5-phosphane (PCl5)
    FI(F)(F)(F)F      -> pentafluoro-lambda5-iodane     (IF5)
    FI(F)F            -> trifluoro-lambda3-iodane        (IF3)

Each emitted name has been confirmed to round-trip through OPSIN 2.9.0 to the
input structure (the λ-convention parses; ``lambda`` ASCII spelling mirrors the
spiro gold ``4lambda4-thiaspiro[3.5]nonane``).

SCOPE (Phase 6, the λ demonstrator — fail-closed, accuracy-first):
  * Hub element restricted to ``_HUB_STEMS`` — chalcogens S/Se/Te, P, and I.
    As/Sb/Bi (Group-15) and the *standard*-valence element hydrides
    (``trichlorophosphane`` PCl3, ``difluorosulfane`` SF2, ``tetrafluorosilane``
    SiF4) are the broader P-68 parent-hydride build and are LEFT for Phase 7/8.
  * Cl/Br are excluded as hubs because RDKit refuses to construct their
    hypervalent forms (``F[Cl](F)F`` is rejected) — only iodine is buildable
    among the halogens.
  * A λ is REQUIRED: if the hub valence is standard (per Table 2.8) this module
    declines and the cascade continues. This keeps Phase 6 to exactly the
    λ-convention capability and never steals a standard-valence element hydride.

This is a graph/atom classifier (NOT a SMARTS broadening — per the
feedback_smarts_and_seniority lesson): every guard below narrows, never widens.
Oxoacids (S/P with O neighbours), oxo-halides (SOCl2), interhalogens (Cl2/ICl),
organo-substituted hubs, di-/poly-nuclear hydrides, ions and rings all fail a
guard and cascade onward — zero false positives.
"""

from typing import Optional

from rdkit import Chem

from ..assembly.naming_utils import get_multiplier_prefix
from .lambda_convention import nonstandard_bonding_number

# Hub element -> parent-hydride stem (P-68 substitutive parent hydrides).
_HUB_STEMS = {
    'S': 'sulfane',
    'Se': 'selane',
    'Te': 'tellane',
    'P': 'phosphane',
    'I': 'iodane',
}

# Halogen substituent prefixes (cited alphanumerically, P-14.5.2; the
# multiplying prefix di/tri/... does NOT count for ordering).
_HALO_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
_HALOGENS = frozenset(_HALO_PREFIX)


def name_mononuclear_hydride(mol) -> Optional[str]:
    """Return the λ-convention PIN for an all-halogen mononuclear hydride with a
    non-standard-valence hub, else ``None`` (fail-closed cascade-continuation).

    Pure: no mol mutation, no global state.
    """
    if mol is None:
        return None

    # Single neutral fragment, no rings, no radicals.
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if mol.GetRingInfo().NumRings() > 0:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != 'H']

    # The hub is the unique nameable central atom (degree >= 2 so diatomic
    # interhalogens like ICl/Cl2 — no central hub — are declined).
    hubs = [a for a in heavy
            if a.GetSymbol() in _HUB_STEMS and a.GetDegree() >= 2]
    if len(hubs) != 1:
        return None
    hub = hubs[0]
    hub_idx = hub.GetIdx()

    # Every other heavy atom must be a terminal halogen single-bonded to the hub.
    halo_counts: dict = {}
    for atom in heavy:
        if atom.GetIdx() == hub_idx:
            continue
        symbol = atom.GetSymbol()
        if symbol not in _HALOGENS or atom.GetDegree() != 1:
            return None
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1 or neighbors[0].GetIdx() != hub_idx:
            return None
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), hub_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
        halo_counts[symbol] = halo_counts.get(symbol, 0) + 1

    if not halo_counts:
        return None
    # The hub's neighbours are exactly the counted halogens (no other element).
    if hub.GetDegree() != sum(halo_counts.values()):
        return None

    # λ is REQUIRED — standard-valence element hydrides are Phase 7/8 scope.
    lam = nonstandard_bonding_number(mol, hub_idx)
    if lam is None:
        return None

    # Assemble: <alphabetical halo prefixes>-lambda<n>-<stem>.
    halo_parts = []
    for symbol in sorted(halo_counts, key=lambda s: _HALO_PREFIX[s]):
        halo_name = _HALO_PREFIX[symbol]
        count = halo_counts[symbol]
        multiplier = get_multiplier_prefix(count, halo_name)
        halo_parts.append(f"{multiplier}{halo_name}")

    stem = _HUB_STEMS[hub.GetSymbol()]
    return f"{''.join(halo_parts)}-lambda{lam}-{stem}"


__all__ = ["name_mononuclear_hydride"]
