"""Dispiro return-arc (segment ``d``) numbering direction — QM9 Class B fix.

 "Linear polyspiro alicyclic ring systems"
(``the Blue Book Blue Book``): the von Baeyer spiro descriptor is cited
"...proceeding consecutively, always by the shorter path, to the other terminal
ring through each spiro atom **and then back to the first spiro atom**." The
first middle-ring arc (descriptor segment ``b``) is numbered forward from the
first spiro atom; the second/return arc (the last segment ``d``) is numbered on
the way BACK, from the second spiro atom to the first — so the return-arc atom
adjacent to the SECOND spiro atom takes the lower locant.

``_dispiro_numbering_candidates`` used to append the return arc in a1->a2 order
(the order ``_find_two_paths`` yields it), i.e. the atom adjacent to the FIRST
spiro atom got the lower locant. Invisible for symmetric middle rings (both arcs
equal) and for all-carbon skeletons (swapping the arcs is a graph automorphism),
it produced a name whose heteroatom / exocyclic-group locant disagreed with its
own descriptor once a heteroatom broke that symmetry. OPSIN then reparsed the
name to a DIFFERENT molecule and the best-effort round-trip gate correctly
abstained ("unknown organic compound").

Concrete: ``C1CC11COC11CCC1`` was numbered ``9-oxadispiro[2.0.3.2]nonane``
(a different molecule) instead of the descriptor-consistent
``8-oxadispiro[2.0.3.2]nonane``.

These are the 10 QM9 Class B witnesses (small O/N dispiro cages, 8 of them with
an exocyclic -one/-ol/-imine/methyl). All must now emit a name that round-trips
to the input by FULL standard InChIKey (best-effort tier; 0-wrong is guaranteed
by the round-trip gate — the assertion here is the stronger RT-CORRECT).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.namer import _validity_gate_name_to_smiles as n2s
from orthonym.rules.spiro import name_spiro_system


# input SMILES -> (a reference name, only for documentation)
WITNESSES = [
    "C1CC11CC=CC11CO1",     # 1-oxadispiro[2.0.2.3]non-8-ene
    "C1CC11COC11CCC1",      # 9-oxadispiro[3.0.2.2]nonane / 8-oxadispiro[2.0.3.2]nonane
    "O=C1CC2(CC2)C11CO1",   # 1-oxadispiro[2.0.2.2]octan-8-one
    "N=C1NC2(CC2)C11CO1",   # 1-oxa-7-azadispiro[2.0.2.2]octan-8-imine
    "O=C1OC2(CC2)C11CO1",   # 1,7-dioxadispiro[2.0.2.2]octan-8-one
    "CC1CC11OCC11CN1",      # 5-methyl-7-oxa-1-azadispiro[2.0.2.2]octane
    "OC1CC11OCC11CN1",      # 7-oxa-1-azadispiro[2.0.2.2]octan-5-ol
    "OC1CC2(CC2)C11CO1",    # 1-oxadispiro[2.0.2.2]octan-8-ol
    "CN1CC11COC11CC1",      # 1-methyl-7-oxa-1-azadispiro[2.0.2.2]octane
    "CC1OC2(CC2)C11CN1",    # 8-methyl-7-oxa-1-azadispiro[2.0.2.2]octane
]


def _inchikey(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m) if m is not None else None


def _besteffort(smi):
    return name_compound(
        smi, general_fallback=True, general_fallback_unverified=True,
        allow_aromatic_general=True)


@pytest.mark.unit
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi", WITNESSES)
def test_class_b_dispiro_cage_roundtrips(smi):
    """Best-effort must EMIT (not abstain) and the name must round-trip to the
    input by full standard InChIKey. return-arc numbering."""
    name = _besteffort(smi)
    assert name and name != "unknown organic compound", (
        f"{smi}: best-effort abstained (return-arc numbering must let it emit)")
    rebuilt = n2s(name)
    assert rebuilt, f"{smi}: emitted name {name!r} did not parse through OPSIN"
    assert _inchikey(rebuilt) == _inchikey(smi), (
        f"{smi}: {name!r} round-trips to a DIFFERENT molecule "
        f"({rebuilt}) — descriptor/locant inconsistency")


@pytest.mark.unit
def test_bare_dispiro_descriptor_consistent_locant():
    """The bare (no exocyclic group) witness pins the exact defect: the O must
    take the descriptor-consistent locant 8, not the old 9 (a different
    molecule). ``[2.0.3.2]`` is the Blue Book descriptor (smaller terminal ring
    first,; low spiro locants,."""
    smi = "C1CC11COC11CCC1"
    name = _besteffort(smi)
    assert name == "8-oxadispiro[2.0.3.2]nonane", name
    assert _inchikey(n2s(name)) == _inchikey(smi)


@pytest.mark.unit
def test_pin_dispiro_return_arc_symmetric_unchanged():
    """Non-regression: the pre-existing all-carbon / symmetric-arc PIN dispiro
    fixtures are BYTE-IDENTICAL — the return-arc reversal is a graph
    automorphism for them (no heteroatom breaks the symmetry), so they must not
    move. These hit the SAME ``_get_polyspiro_numbering`` this fix touches."""
    cases = {
        "C1CC12CC1(CC1)C2": "dispiro[2.1.2.1]octane",
        "C1CCCC12CCC1(CCCC1)CC2": "dispiro[4.2.4.2]tetradecane",
        "C1CC12C1(CC1)C2": "dispiro[2.0.2.1]heptane",
        # CQ5 hetero witnesses (round-1 first-arc fix) must stay put too.
        "C1C2(CCC2)C11CCO1": "1-oxadispiro[3.0.3.1]nonane",
        "C1CC11CCC11CO1": "1-oxadispiro[2.0.2.2]octane",
    }
    for smi, expected in cases.items():
        got = name_spiro_system(Chem.MolFromSmiles(smi))[0]
        assert got == expected, f"{smi}: {got!r} != {expected!r}"
