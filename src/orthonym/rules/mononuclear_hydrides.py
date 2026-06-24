"""Mononuclear parent-hydride namer (P-68 / P-21.1 / P-31.1.4.2 λ-convention).

Names a single non-carbon "hub" atom — a Group-15 pnictogen (P/As/Sb/Bi), a
chalcogen (S/Se/Te) or iodine — as a *substitutive parent hydride* (P-68.3),
optionally bearing the λ-convention when the hub valence is non-standard::

    FS(F)(F)(F)(F)F   -> hexafluoro-lambda6-sulfane    (SF6, λ6 nonstandard)
    FS(F)(F)F         -> tetrafluoro-lambda4-sulfane    (SF4, λ4)
    FP(F)(F)(F)F      -> pentafluoro-lambda5-phosphane  (PF5, λ5)
    FI(F)(F)(F)F      -> pentafluoro-lambda5-iodane      (IF5, λ5)
    ClP(Cl)Cl         -> trichlorophosphane              (PCl3, standard valence)
    FS(F)             -> difluorosulfane                 (SF2, standard valence)
    Cl[As](Cl)Cl      -> trichloroarsane                 (AsCl3, Group-15)
    C[As](C)C         -> trimethylarsane                 (organyl As, Group-15)
    c1ccccc1[As]...   -> triphenylarsane
    [AsH3]            -> arsane                           (bare parent hydride)

Every emitted name round-trips through OPSIN 2.9.0 to the input structure (the
λ-convention parses; ``lambda`` ASCII spelling mirrors the spiro gold
``4lambda4-thiaspiro[3.5]nonane``).

SCOPE (fail-closed, accuracy-first):
  * Two substituent regimes, never mixed (a mixed halo+organyl hub fails both
    guards and cascades onward — fail-closed):
      - **All-halogen** hub (any element in ``_HUB_STEMS``, hub degree >= 2):
        the halogen-prefix block (no enclosing marks — halogens are simple
        substituents, P-16.3.3) + optional λ. Covers SF6/SF4/PF5/IF5 (λ) AND the
        standard-valence PCl3/PF3/SF2/AsCl3/SbBr3/BiCl3 (no λ).
      - **Organyl / bare** hub restricted to **Group-15 As/Sb/Bi only**
        (``_ORGANYL_HUBS``): every substituent a pure unbranched-alkyl or
        phenyl/naphthyl organyl, named via the mononuclear-parent-hydride
        enclosing-mark rule (P-16.5.1.3). Covers trimethylarsane / triphenylarsane
        and the bare arsane/stibane/bismuthane.
  * P / S / Se / Te / I take only the all-halogen regime here. Carbon-substituted
    P (trimethylphosphane) stays with ``rules.phosphorus.name_phosphine`` (no
    double-claim); carbon-substituted chalcogens are sulfides/sulfanes named
    elsewhere; the organyl regime is reserved for the Group-15 metals As/Sb/Bi,
    which otherwise route to the P-69 organometallic handler and are mis-opened to
    a carbon chain (``C[As](C)C`` -> ``(methylmethyl)methane``).
  * Cl/Br are never hubs (RDKit refuses to construct their hypervalent forms);
    only iodine builds among the halogens, and interhalogens (ICl, degree 1) are
    excluded by the hub-degree>=2 guard.

This is a graph/atom classifier (NOT a SMARTS broadening — per the
feedback_smarts_and_seniority lesson): every guard narrows, never widens.
Oxoacids (hub with O neighbours), oxo-halides (SOCl2), organo-substituted
chalcogens, di-/poly-nuclear hydrides (diphosphane), ions and rings all fail a
guard and cascade onward — zero false positives.
"""

from typing import List, Optional

from rdkit import Chem

from ..assembly.naming_utils import get_multiplier_prefix
from .lambda_convention import nonstandard_bonding_number
from .phosphorus import _build_substituent_string
from .substituent_purity import pure_organyl_prefix_name

# Hub element -> parent-hydride stem (P-68 / P-21.1 substitutive parent hydrides).
_HUB_STEMS = {
    'S': 'sulfane',
    'Se': 'selane',
    'Te': 'tellane',
    'P': 'phosphane',
    'As': 'arsane',
    'Sb': 'stibane',
    'Bi': 'bismuthane',
    'I': 'iodane',
}

# Hubs that additionally accept ORGANYL / bare substituents (no pre-existing
# substitutive namer; the P-69 organometallic handler mangles them). Restricted
# to the Group-15 metals; P stays with name_phosphine, chalcogens/iodine with
# their halide-only regime.
_ORGANYL_HUBS = frozenset({'As', 'Sb', 'Bi'})

# Halogen substituent prefixes (cited alphanumerically, P-14.5.2; the
# multiplying prefix di/tri/... does NOT count for ordering).
_HALO_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
_HALOGENS = frozenset(_HALO_PREFIX)


def _find_unique_hub(mol):
    """Return the unique nameable hub atom, or None.

    A nameable hub is the SOLE heavy atom whose element is in ``_HUB_STEMS``
    (so diphosphane PH2-PH2 = two P hubs -> None; an all-halogen molecule with no
    hub -> None)."""
    hubs = [a for a in mol.GetAtoms()
            if a.GetSymbol() != 'H' and a.GetSymbol() in _HUB_STEMS]
    return hubs[0] if len(hubs) == 1 else None


def _classify_halogens(mol, hub) -> Optional[dict]:
    """Return ``{halogen_symbol: count}`` iff EVERY non-hub heavy atom is a
    terminal halogen single-bonded to the hub, else None."""
    hub_idx = hub.GetIdx()
    halo_counts: dict = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H' or atom.GetIdx() == hub_idx:
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
    # The hub's neighbours are exactly the counted halogens (no other element).
    if not halo_counts or hub.GetDegree() != sum(halo_counts.values()):
        return None
    return halo_counts


def _classify_organyls(mol, hub) -> Optional[List[str]]:
    """Return the list of organyl prefix names (possibly empty for a bare hydride)
    iff EVERY heavy hub-neighbour is a valid pure-hydrocarbyl organyl AND no stray
    heteroatom exists, else None.

    A halogen neighbour (mixed halo+organyl) -> None (fail-closed; the all-halogen
    regime is handled separately)."""
    hub_idx = hub.GetIdx()
    names: List[str] = []
    for nbr in hub.GetNeighbors():
        if nbr.GetSymbol() == 'H':
            continue
        if nbr.GetSymbol() in _HALOGENS:
            return None  # mixed halo+organyl -> fail-closed
        name = pure_organyl_prefix_name(mol, nbr.GetIdx(), hub_idx)
        if name is None:
            return None
        names.append(name)
    # Defensive: every heavy atom is the hub or a carbon (the per-neighbour purity
    # walk rejects any heteroatom inside a substituent; a single connected
    # fragment then guarantees there is no stray heteroatom elsewhere).
    if any(a.GetSymbol() != 'C' and a.GetIdx() != hub_idx
           for a in mol.GetAtoms() if a.GetSymbol() != 'H'):
        return None
    return names


def _assemble(prefix_block: str, lam: Optional[int], stem: str) -> str:
    """Compose ``<prefix><stem>`` (standard valence) or
    ``<prefix>-lambda<n>-<stem>`` (non-standard valence, P-31.1.4.2)."""
    if lam is None:
        return f"{prefix_block}{stem}"
    if not prefix_block:
        return f"lambda{lam}-{stem}"
    return f"{prefix_block}-lambda{lam}-{stem}"


def name_mononuclear_hydride(mol) -> Optional[str]:
    """Return the substitutive PIN for a mononuclear parent hydride (P-68 /
    P-31.1.4.2), else ``None`` (fail-closed cascade-continuation).

    Pure: no mol mutation, no global state.
    """
    if mol is None:
        return None

    # Single neutral fragment, no radicals.
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    hub = _find_unique_hub(mol)
    if hub is None:
        return None
    # The hub itself must NOT be a ring member (an arsenin/thiophene/arsole ring
    # parent is a heterocycle, not a mononuclear parent hydride) — but aromatic
    # ring SUBSTITUENTS (triphenylarsane) are fine: the per-substituent purity
    # guard handles those, so we check the hub specifically, not the whole mol.
    if hub.IsInRing():
        return None
    hub_idx = hub.GetIdx()
    stem = _HUB_STEMS[hub.GetSymbol()]
    lam = nonstandard_bonding_number(mol, hub_idx)

    # --- All-halogen regime (every hub element; hub degree >= 2 excludes the
    #     diatomic interhalogens ICl/IBr). Halogens are simple substituents:
    #     alphanumerical concatenation with NO enclosing marks (P-16.3.3). ---
    halo_counts = _classify_halogens(mol, hub)
    if halo_counts is not None and hub.GetDegree() >= 2:
        halo_parts = []
        for symbol in sorted(halo_counts, key=lambda s: _HALO_PREFIX[s]):
            halo_name = _HALO_PREFIX[symbol]
            multiplier = get_multiplier_prefix(halo_counts[symbol], halo_name)
            halo_parts.append(f"{multiplier}{halo_name}")
        return _assemble(''.join(halo_parts), lam, stem)

    # --- Organyl / bare regime (Group-15 As/Sb/Bi only). ---
    if hub.GetSymbol() in _ORGANYL_HUBS:
        organyls = _classify_organyls(mol, hub)
        if organyls is not None:
            prefix_block = _build_substituent_string(organyls) if organyls else ""
            return _assemble(prefix_block, lam, stem)

    return None


__all__ = ["name_mononuclear_hydride"]
