"""Universal coverage-floor: skeleton-membership ALLOW-LIST (M1 lever).

Only carbon and a ``REPLACEMENT_TERMS``-spellable heteroatom MAY thread the
chain spine (``_tree_neighbors`` -> ``_is_skeletal_spine_element``). This
generalizes the terminal-halogen exclusion (commit d3590326) to the WHOLE
non-skeletal class in ONE closed check, derived directly from the same table
``_build_hetero_prefix`` uses to spell a skeletal heteroatom.

Root cause (spiro-followup Task A cascade-spy, 2026-08-30): a non-skeletal atom
threaded into the chain spine was COUNTED in the chain length (a phantom carbon)
yet SILENTLY SKIPPED by ``_build_hetero_prefix`` (``if sym in
REPLACEMENT_TERMS``) -- i.e. dropped, yielding a coverage-complete name of a
DIFFERENT constitution (CHF2 -> ``1-fluoroethan-1-yl``: extra C, dropped F).
The terminal halogen was one instance; a hypervalent/exotic-valence halogen
(degree >= 2, which the old degree-1 ``_is_terminal_halogen`` never caught),
a metal, a noble gas and Al/Ga/In/Tl are others.

0-WRONG-CRITICAL invariant: the allow-list is tied to ``REPLACEMENT_TERMS`` BY
CONSTRUCTION, so it can never drop a legitimate skeletal heteroatom (which would
regress a correct name) -- membership == spellability. Every skeletal-
replacement element is tested IN; every non-skeletal shape is tested OUT; and
the legitimate skeletal-heteroatom chains (ether/thioether/aza/silane/borane/
stannane) are asserted to still round-trip.

Reachable ONLY on the best-effort coverage floor (this whole module is
``_general_fallback``-gated in namer.py), so PIN/default output is byte-identical
-- proven by the fast gate. IUPAC 2013 P-15.4 / P-21 ('a'-replacement skeletal
element set); P-29.3 / P-35.1 (halogen substituent prefixes).

Targeted-file run only (avoid the OPSIN-pipe deadlock of a full pytest run):
    .venv/bin/python -m pytest tests/unit/assembly/test_universal_skeleton_allowlist.py -q
"""
from __future__ import annotations

from rdkit import Chem

from orthonym.assembly.universal_substituent import (
    _SKELETAL_SPINE_ELEMENTS,
    _is_skeletal_spine_element,
    name_universal_substitutive,
)
from orthonym.rules.skeletal_replacement import REPLACEMENT_TERMS
from orthonym.validation.reconstruct import verify_or_none


def _heavy(smiles):
    m = Chem.MolFromSmiles(smiles)
    return frozenset(a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1)


# ---------------------------------------------------------------------------
# Membership -- exhaustive, both directions.
# ---------------------------------------------------------------------------

def test_allowlist_is_exactly_carbon_plus_replacement_terms():
    """The 0-wrong-critical invariant: the spine allow-list is EXACTLY the set
    ``_build_hetero_prefix`` can spell (carbon + REPLACEMENT_TERMS). If this
    ever drifts, a spellable heteroatom could be dropped (regression) or an
    unspellable one absorbed as a phantom carbon (wrong constitution)."""
    assert _SKELETAL_SPINE_ELEMENTS == frozenset({"C"}) | frozenset(REPLACEMENT_TERMS)


def test_every_skeletal_replacement_element_is_allowed():
    """Carbon + every 'a'-replacement heteroatom the chain machinery can name
    MUST be allowed onto the spine (else its correct name regresses)."""
    assert _is_skeletal_spine_element_by_symbol("C")
    for el in REPLACEMENT_TERMS:  # O S Se Te N P As Sb Bi Si Ge Sn Pb B
        assert _is_skeletal_spine_element_by_symbol(el), (
            f"{el} is spellable by _build_hetero_prefix but not allow-listed")


def test_non_skeletal_elements_are_excluded():
    """Halogens, metals, noble gases and the off-table 'a'-terminators
    Al/Ga/In/Tl (no REPLACEMENT_TERMS morpheme) MUST be kept off the spine --
    threading them absorbs a phantom carbon and silently drops the atom."""
    non_skeletal = [
        "F", "Cl", "Br", "I", "At",          # halogens (P-35.1 prefixes)
        "Al", "Ga", "In", "Tl",               # off-table 'a'-terminators
        "Fe", "Na", "K", "Mg", "Zn", "Cu",    # metals (organometallic, P-69, OOS)
        "He", "Ne", "Ar",                     # noble gases
    ]
    for el in non_skeletal:
        assert not _is_skeletal_spine_element_by_symbol(el), (
            f"{el} is not spellable as a skeletal atom but was allow-listed")


def _is_skeletal_spine_element_by_symbol(sym: str) -> bool:
    """Build a single-atom mol of *sym* and query the predicate on idx 0."""
    m = Chem.MolFromSmiles(f"[{sym}]")
    assert m is not None, f"could not build a probe atom for {sym!r}"
    return _is_skeletal_spine_element(m, 0)


# ---------------------------------------------------------------------------
# Regression -- legitimate skeletal heteroatoms STILL thread and round-trip.
# (The allow-list must not drop any of these -- 0-wrong-critical.)
# ---------------------------------------------------------------------------

def _names_and_roundtrips(smiles):
    m = Chem.MolFromSmiles(smiles)
    r = name_universal_substitutive(m)
    assert r is not None, f"expected a name for {smiles!r}, got None"
    assert r.covers == _heavy(smiles), (
        f"coverage gap for {smiles!r}: covers={r.covers} heavy={_heavy(smiles)}")
    assert verify_or_none(r.name, smiles) == r.name, (
        f"{r.name!r} did not round-trip for {smiles!r}")
    return r


def test_ether_chain_threads_oxa():
    r = _names_and_roundtrips("CCOCCOCC")  # 3,6-dioxaoctane
    assert "oxa" in r.name

def test_thioether_threads_thia():
    r = _names_and_roundtrips("CSC")       # 2-thiapropane
    assert "thia" in r.name

def test_aza_chain_threads_aza():
    r = _names_and_roundtrips("CCNCC")     # 3-azapentane
    assert "aza" in r.name

def test_silane_threads_sila():
    r = _names_and_roundtrips("C[Si](C)(C)C")  # 2-silapropane skeleton
    assert "sila" in r.name

def test_borane_threads_bora():
    r = _names_and_roundtrips("CCB(CC)CC")     # 3-borapentane skeleton
    assert "bora" in r.name

def test_stannane_threads_stanna():
    r = _names_and_roundtrips("C[Sn](C)(C)C")
    assert "stanna" in r.name


# ---------------------------------------------------------------------------
# Behavior -- a non-skeletal atom is NEVER absorbed as a phantom carbon that
# round-trips. It is either rendered as its own substituent leaf (halogen) OR
# the whole call fails closed (None / RT-fail), so the offer RT gate abstains.
# ---------------------------------------------------------------------------

def test_terminal_halogens_render_as_leaf_prefix():
    for smi, tok in (("CCCCl", "chloro"), ("CCCCBr", "bromo"), ("CCCI", "iodo")):
        r = _names_and_roundtrips(smi)
        assert tok in r.name, f"{smi} expected {tok}: {r.name!r}"


def test_hypervalent_halogen_never_ships_a_phantom_carbon_name():
    """A degree-3 iodine(III) -- which the OLD degree-1 terminal-halogen check
    never excluded -- must NOT yield a coverage-complete name that round-trips
    to a DIFFERENT constitution (the phantom-carbon hazard). The allow-list
    keeps it off the spine; whatever the floor builds must FAIL the RT gate
    (or void), so the caller abstains rather than shipping a wrong molecule."""
    smi = "c1ccccc1[I](Cl)Cl"
    m = Chem.MolFromSmiles(smi)
    r = name_universal_substitutive(m)
    # If a name is built at all, it must NOT round-trip (the I cannot be a
    # skeletal carbon), so the offer RT gate voids it -> abstain, 0-wrong.
    if r is not None:
        assert verify_or_none(r.name, smi) != r.name, (
            f"phantom-carbon name for a hypervalent halogen round-tripped: "
            f"{r.name!r}")


def test_metal_center_fails_closed_not_phantom_carbon():
    """An organometallic centre (P-69, out of scope) must fail closed -- void
    or RT-fail -- never a phantom-carbon name that round-trips."""
    smi = "CC[Fe]CC"
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return  # RDKit rejected the valence; nothing to name (still 0-wrong)
    r = name_universal_substitutive(m)
    if r is not None:
        assert verify_or_none(r.name, smi) != r.name, (
            f"phantom-carbon name for a metal centre round-tripped: {r.name!r}")


def test_ylidene_halogen_fails_closed_not_phantom_carbon():
    """A ``C=I`` iodine ylidene (double-bonded I, matched by no _LEAF_DOUBLE
    leaf) must NOT fall through to a 1-atom chain and render a phantom
    ``methane`` -- the builder-side allow-list guard voids it. (Was a
    raw-core WRONG_CONSTITUTION on base HEAD; now a clean abstain.)"""
    smi = "C=Ic1ccccc1"
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return
    r = name_universal_substitutive(m)
    if r is not None:
        assert verify_or_none(r.name, smi) != r.name, (
            f"phantom-carbon name for a C=I ylidene round-tripped: {r.name!r}")


# ---------------------------------------------------------------------------
# Ring-builder side: a ring is spelled through the SAME _build_hetero_prefix,
# so a non-skeletal RING atom must fail closed too (symmetric to the chain).
# ---------------------------------------------------------------------------

def test_skeletal_hetero_rings_still_build():
    """Legitimate monocyclic heterocycles (skeletal ring heteroatoms) must
    still be named and round-trip -- the ring guard must not drop them."""
    for smi, tok in (("O1CCCCC1", "oxa"), ("S1CCCCC1", "thia"), ("N1CCCCC1", "aza")):
        r = _names_and_roundtrips(smi)
        assert tok in r.name, f"{smi} expected {tok}: {r.name!r}"


def test_iodinane_ring_fails_closed_not_phantom_carbocycle():
    """An iodinane / λ-halane ring (a non-skeletal HALOGEN as a ring atom,
    ``I1CCCCC1``) has no REPLACEMENT_TERMS morpheme, so _build_hetero_prefix
    would SILENTLY drop it and render a phantom carbocycle. The ring-builder
    allow-list guard voids instead -> abstain, never a wrong constitution."""
    smi = "C1CCC[IH]C1"
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return  # RDKit rejected the hypervalent iodine; nothing to name
    r = name_universal_substitutive(m)
    if r is not None:
        assert verify_or_none(r.name, smi) != r.name, (
            f"phantom-carbocycle name for an iodinane ring round-tripped: "
            f"{r.name!r}")
