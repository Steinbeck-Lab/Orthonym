"""Inositol (cyclitol) retained names —.

 a phase follow-on (Blue Book audit unit p104-106, finding F1).

The nine stereoisomers of cyclohexane-1,2,3,4,5,6-hexol each have a retained
italic-prefix name (``myo-``, ``scyllo-``, ``cis-``, ``epi-``, ``neo-``,
``allo-``, ``muco-``, ``D-chiro-``, ``L-chiro-``) that is the PREFERRED IUPAC
name.

These names are **OPSIN-unparseable** (verified: ``OpsinOracle.name_to_smiles``
returns ``None`` for every ``<prefix>-inositol``), so there is NO round-trip
oracle for them — this is a NAME-EXACT recogniser (the oracle-blind / name-exact
validation tier).

Correctness is grounded two ways so the absence of an RT oracle is not a hole:

* **Structures** are RDKit-enumerated — ``EnumerateStereoisomers`` on the flat
  hexol yields EXACTLY nine distinct stereoisomers (the nine inositols), so the
  table is provably complete and every key is a real, canonical structure.
* **InChIKey -> name** labels are cross-verified against authoritative sources:
  NIST WebBook (myo / scyllo / muco / epi / allo / chiro by InChIKey search),
  Wikidata + the chemical literature for the D/L-chiro assignment, cis / neo
  (the two highest-symmetry meso isomers: cis is all-cis, the only all-carbons-
  equivalent isomer besides all-trans scyllo). myo-inositol's key
  ``an InChIKey`` matches NIST CAS 87-89-8.

Fail-closed: ``name_inositol`` returns ``None`` for anything that is not
EXACTLY one of the nine fully-stereodefined cyclohexanehexols. An undefined- or
partial-stereo hexol (flat InChIKey ``an InChIKey``) is not in
the table, so it keeps the systematic ``cyclohexane-1,2,3,4,5,6-hexol`` name; a
substituted/deoxy/larger ring never reaches the table.
"""

from typing import Optional

from rdkit.Chem import inchi

# Standard-InChIKey -> retained PIN. The shared skeleton block is
# CDAISMWEOUEBRE; the nine distinct stereo blocks are the nine inositols.
_INOSITOL_BY_INCHIKEY = {
    # The seven ACHIRAL (meso) inositols — RDKit perceives their stereochemistry
    # deterministically across atom orderings (the standard InChIKey is stable),
    # so they ship name-exact and pass the determinism gate.
    "CDAISMWEOUEBRE-GPIVLXJGSA-N": "myo-inositol",      # NIST CAS 87-89-8
    "CDAISMWEOUEBRE-CDRYSYESSA-N": "scyllo-inositol",   # NIST
    "CDAISMWEOUEBRE-GNIYUCBRSA-N": "muco-inositol",     # NIST
    "CDAISMWEOUEBRE-NIPYSYMMSA-N": "epi-inositol",      # NIST
    "CDAISMWEOUEBRE-OQYPVSDDSA-N": "allo-inositol",     # NIST
    "CDAISMWEOUEBRE-JMVOWJSSSA-N": "cis-inositol",      # all-cis (lit. + symmetry)
    "CDAISMWEOUEBRE-DCLYFUHFSA-N": "neo-inositol",      # lit. + elimination
    # --- DELIBERATELY EXCLUDED: the chiral chiro pair (DETERMINISM, not data) ---
    # D-chiro-inositol (…-LKPKBOIGSA-N, Wikidata Q3011024 + NIST) and its
    # enantiomer L-chiro-inositol (…-SHFUYGGZSA-N) are the only CHIRAL inositols.
    # RDKit's stereo perception of this pseudo-C2-symmetric chiral pair is
    # ORDER-DEPENDENT: the same molecule round-trips to either enantiomer's
    # standard InChIKey depending on SMILES atom order (verified — flips D<->L
    # across random spellings, even through Chem.CanonSmiles). Cataloguing them
    # would ship the WRONG enantiomer ~50% of the time AND fail the determinism
    # gate. They therefore stay FAIL-CLOSED (the chiral systematic CIP name is
    # likewise order-dependent and OPSIN-suppressed -> a deterministic 'unknown').
    # Revisit if/when RDKit stabilises chiral perception for symmetric polyols.
}

# The nine retained inositol PIN strings. The OPSIN validity gate consults this
# to UN-SUPPRESS them: they are OPSIN-unparseable (generation-grammar gap) but
# correct-by-construction (the InChIKey recogniser produces them only for a
# verified inositol structure), so they ship name-exact (cf. the thioperoxol
# carve-out). A frozenset for O(1) gate-path membership.
INOSITOL_NAMES = frozenset(_INOSITOL_BY_INCHIKEY.values())


def is_inositol_skeleton(mol) -> bool:
    """True iff ``mol`` is exactly cyclohexane-1,2,3,4,5,6-hexol.

    Requires: C6H12O6 with a SINGLE all-carbon six-membered ring, each ring
    carbon bearing exactly one hydroxyl (a degree-1 O with one H) and one H, and
    no other heavy atoms. This hard gate keeps the InChIKey lookup off every
    non-cyclitol molecule and fail-closes on deoxy / O-substituted / phosphate /
    larger-ring relatives.
    """
    if mol is None:
        return False
    atoms = mol.GetAtoms()
    if mol.GetNumAtoms() != 12:  # 6 C + 6 O (hydrogens implicit)
        return False
    if sum(1 for a in atoms if a.GetSymbol() == 'C') != 6:
        return False
    if sum(1 for a in atoms if a.GetSymbol() == 'O') != 6:
        return False

    ri = mol.GetRingInfo()
    if ri.NumRings() != 1:
        return False
    rings = ri.AtomRings()
    if len(rings) != 1 or len(rings[0]) != 6:
        return False
    ring = set(rings[0])
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring):
        return False

    for i in ring:
        a = mol.GetAtomWithIdx(i)
        if a.GetTotalNumHs() != 1:          # one H on the ring carbon
            return False
        o_nbrs = [n for n in a.GetNeighbors() if n.GetSymbol() == 'O']
        if len(o_nbrs) != 1:                # exactly one hydroxyl
            return False
        o = o_nbrs[0]
        if o.GetDegree() != 1 or o.GetTotalNumHs() != 1:  # terminal -OH
            return False
    return True


def name_inositol(mol) -> Optional[str]:
    """Return the retained inositol PIN for a fully-stereodefined
    cyclohexanehexol matching one of the seven deterministically-perceived meso
    inositols, else ``None``.

    Fail-closed: a non-cyclitol, the chiral chiro pair (see
    ``is_chiral_inositol``), or a cyclohexanehexol whose stereochemistry is
    undefined/partial (standard InChIKey carries no stereo layer) is not in the
    table and returns ``None`` (the undefined-stereo hexol then keeps its
    systematic name).
    """
    if not is_inositol_skeleton(mol):
        return None
    try:
        ik = inchi.MolToInchiKey(mol)
    except Exception:
        return None
    return _INOSITOL_BY_INCHIKEY.get(ik)


def is_chiral_inositol(mol) -> bool:
    """True iff ``mol`` is a fully-stereodefined cyclohexanehexol that is NOT one
    of the seven catalogued meso inositols — i.e. the chiral D-/L-chiro-inositol
    pair.

    RDKit perceives this pseudo-C2-symmetric CHIRAL pair NON-DETERMINISTICALLY:
    the standard InChIKey flips D<->L by SMILES atom order (verified), so the
    molecule cannot be named (or systematically CIP-labelled) reproducibly.
    Naming must therefore REFUSE it (a deterministic 'unknown') rather than ship
    a flipping wrong-enantiomer name. The test is order-stable even though the
    InChIKey is not: ``is_inositol_skeleton`` is stereo-independent, and "has a
    stereo layer but is not a catalogued meso inositol" is true for BOTH chiro
    enantiomers regardless of which way RDKit flipped this evaluation.

    The flat / undefined-stereo hexol (InChIKey stereo block ``UHFFFAOYSA``) is
    NOT a chiral inositol and returns ``False`` so it keeps the systematic name.
    """
    if not is_inositol_skeleton(mol):
        return False
    try:
        ik = inchi.MolToInchiKey(mol)
    except Exception:
        return False
    if ik in _INOSITOL_BY_INCHIKEY:      # a deterministic meso inositol
        return False
    parts = ik.split('-')
    stereo_block = parts[1] if len(parts) >= 2 else 'UHFFFAOYSA'
    return stereo_block != 'UHFFFAOYSA'  # defined stereo + not meso -> chiro
