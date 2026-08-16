"""P-63.3.1 / P-63.3.2 divalent-chalcogen bridge substituent prefixes.

The defect these tests lock down was a SILENT ATOM DROP, the worst failure mode
this system has: a branch that no producer could name was skipped and the rest of
the molecule was named anyway, so the emitted name described fewer atoms than
were drawn.

  CC(C)(C)OOCCO   9 heavy atoms  ->  'ethan-1-ol'   (3 heavy; SIX atoms gone)
  CC(C)OOCCO      8 heavy atoms  ->  'ethan-1-ol'   (a DIFFERENT molecule, SAME name)
  CC(C)(C)SSCCO                  ->  '2-disulfanediyl-2-(tert-butyldisulfanyl)ethan-1-ol'
                                     (the S-S counted TWICE, and 'disulfanediyl' is
                                      the MULTIPLICATIVE divalent bridge, P-63.3.1(3))
  CC(C)(C)OSCCO                  ->  '2-[(2-hydroxy-2-methylpropyl)sulfanyl]ethan-1-ol'
                                     (the O-S bridge REARRANGED into a C-S bond)

The central test here is an INVARIANT over a generated family, not a check of
those four strings: no emitted name may describe fewer heavy atoms than the input
molecule, and any name that is emitted must describe THAT molecule. Abstention is
always an acceptable answer; a wrong name never is.

These tests assert RAW PRODUCER output. The autouse fixture at
tests/conftest.py:266 disables the OPSIN validity gate for every test, so nothing
here is rescued by the gate re-picking a candidate that happens to parse. That is
deliberate and it is the only honest way to test this class: the gate cannot fire
at all when no OPSIN jar is present, which is a supported mode, so a producer that
is wrong-but-gate-rescued still ships wrong names to some users. Everything below
therefore holds the producers themselves to the Blue Book.

Blue Book authority (verified on disk in BlueBookV2/BlueBookV2.md and against the
online edition at https://iupac.qmul.ac.uk/BlueBook/):
  * P-15.3.1.2.1.1 (BB 5194)  -OO- 'peroxy' preselected; 'dioxy' abolished.
  * P-63.3.1 (BB 27858-27860) method (1) is substitutive and gives the PIN:
    R'-peroxy / R'-disulfanyl. 'disulfanediyl' is method (3), the MULTIPLICATIVE
    divalent bridge -SS-, licensed only when both ends are parent structures.
  * P-51.4.1.1 (BB 23389)  '(tert-butylperoxy)dimethylsilyl propanoate (PIN)'
    -- 'tert-butyl' is cited BARE inside the compound prefix, no inner marks.
  * P-63.2.2.2 (BB 27665-27691)  retained R-O- contractions: 'tert-butoxy
    (preferred prefix) (no substitution)', explicitly not 'tert-butyloxy';
    '(propan-2-yl)oxy' is the preferred prefix, 'isopropoxy' general-only.
  * P-57.1.3 (BB 24426)  'isopropyl' is barred from preferred IUPAC names.
  * P-63.3.2 (BB 27914)  '(methoxysulfanyl)cyclohexane (PIN)' for C6H11-S-O-CH3
    and '[(methylsulfanyl)oxy]ethane (PIN)' for CH3-CH2-O-S-CH3: the LAST-cited
    chalcogen is the atom bonded to the parent.
  * P-71.3 (BB 40737)  'tert-butyldisulfanyl (PIN)'.
  * P-71.2.1.2  'tert-butoxytri(phenyl)-λ5-phosphanyl (PIN)' -- tert-butoxy
    concatenated BARE inside a compound prefix of a PIN, with the bracketed
    '[(2-methylpropan-2-yl)oxy]...' variant listed as NOT preferred. This is what
    establishes 'tert-butoxysulfanyl' rather than leaving it a guess.
  * P-41 Table 4.1 (BB 18190/18218)  hydroxy = class 17, peroxides = class 42, so
    P-44.1.1 (BB 18875) picks the ethanol parent in every case below.
  * P-14.3.4.2(b) + P-14.3.3  the locant '1' may be elided only for a
    MONOsubstituted two-atom chain, so these disubstituted C2 parents are
    'ethan-1-ol', never 'ethanol'.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import (
    _name_mixed_chalcogen_branch,
    composed_chalcogen_group_prefix,
    is_dichalcogen_bridge_attach,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OPSIN_JAR = PROJECT_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"


# ---------------------------------------------------------------------------
# The generated family. Every organyl R is crossed with every bridge and every
# parent, so the invariant below is asserted over a family and not over the four
# molecules that were reported.
# ---------------------------------------------------------------------------

# R group SMILES fragment, written so that "<R><bridge><parent>" is valid SMILES.
_R_GROUPS = (
    "C",             # methyl
    "CC",            # ethyl
    "CCC",           # propyl
    "CC(C)",         # propan-2-yl  (isopropyl is barred from PINs, P-57.1.3)
    "CC(C)(C)",      # tert-butyl
    "OCC",           # 2-hydroxyethyl
)

# ARYL organyl roots are left out of the -SS-/-OS- family on purpose. P-63.3.1 draws
# no distinction between an alkyl and an aryl R, and '<aryl>disulfanyl' IS a compound
# prefix that must be enclosed (bare, '2-phenyldisulfanylethan-1-ol' re-parses as
# '2-phenyl' + 'disulfanyl' -> a different molecule). But that enclosure is decided
# by the shared naming_utils root rule consulted by the caller, not by the producers
# in this module, so asserting it here would be testing another module's rule
# through this one. The aryl root IS exercised below via the -OO- bridge, whose
# 'peroxy' clause in that predicate is long-standing and independent.

# Divalent-chalcogen bridges. P-63.3.1 homo bridges and P-63.3.2 mixed ones.
_BRIDGES = ("OO", "SS", "OS", "SO")

# Parents whose principal characteristic group outranks the bridge class, so the
# bridge is always expressed as a PREFIX (P-41 Table 4.1: 17 and 12 beat 42).
_PARENTS = (
    "CCO",           # ethan-1-ol
    "CCC(=O)O",      # propanoic acid
)


def _family(r_groups=_R_GROUPS):
    """Every (smiles, canonical, heavy_atom_count) in the generated bridge family."""
    seen = set()
    for r in r_groups:
        for bridge in _BRIDGES:
            for parent in _PARENTS:
                smiles = f"{r}{bridge}{parent}"
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                canonical = Chem.CanonSmiles(smiles)
                if canonical in seen:
                    continue
                seen.add(canonical)
                yield smiles, canonical, mol.GetNumHeavyAtoms()


def _is_abstention(name) -> bool:
    """True when the namer refused rather than emitted a name."""
    if not name:
        return True
    lowered = name.strip().lower()
    return (
        lowered.startswith("unknown")
        or lowered.startswith("unsupported")
        or "unable" in lowered
    )


def _name(smiles):
    from orthonym.namer import name_compound

    try:
        return name_compound(smiles)
    except Exception:
        return None


def _opsin_batch(names):
    """Parse many names through ONE OPSIN JVM -> list of SMILES-or-None."""
    proc = subprocess.run(
        ["java", "-jar", str(OPSIN_JAR), "-osmi"],
        input="\n".join(names) + "\n",
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = proc.stdout.split("\n")
    out = []
    for i in range(len(names)):
        raw = lines[i].strip() if i < len(lines) else ""
        out.append(raw or None)
    return out


# ---------------------------------------------------------------------------
# THE INVARIANT
# ---------------------------------------------------------------------------


def _assert_family_invariant(family, min_size):
    """No member of ``family`` may be named with fewer heavy atoms than it has,
    and any name emitted must describe THAT molecule. Shared by the in-territory
    invariant test and its xfail-marked aryl twin so both assert the same rule."""
    assert len(family) >= min_size, (
        f"family too small to be an invariant test: {len(family)}"
    )

    emitted = [(s, c, h, _name(s)) for s, c, h in family]
    named = [(s, c, h, n) for s, c, h, n in emitted if not _is_abstention(n)]
    assert named, "every member abstained -- the invariant would be vacuous"

    parsed = _opsin_batch([n for _, _, _, n in named])

    dropped = []
    wrong = []
    for (smiles, canonical, heavy, name), opsin_smiles in zip(named, parsed):
        if opsin_smiles is None:
            # An unparseable name cannot be shown to describe the molecule.
            wrong.append((smiles, name, "OPSIN-UNPARSEABLE"))
            continue
        opsin_mol = Chem.MolFromSmiles(opsin_smiles)
        if opsin_mol is None:
            wrong.append((smiles, name, "RDKit rejected OPSIN output"))
            continue
        if opsin_mol.GetNumHeavyAtoms() < heavy:
            dropped.append(
                (smiles, name, heavy, opsin_mol.GetNumHeavyAtoms())
            )
            continue
        if Chem.CanonSmiles(opsin_smiles) != canonical:
            wrong.append((smiles, name, Chem.CanonSmiles(opsin_smiles)))

    assert not dropped, (
        "ATOM DROP -- these names describe fewer heavy atoms than the input:\n"
        + "\n".join(
            f"  {s}: {h_in} heavy -> {n!r} = {h_out} heavy" for s, n, h_in, h_out in dropped
        )
    )
    assert not wrong, (
        "these names do not describe the molecule they were generated for:\n"
        + "\n".join(f"  {s}: {n!r} -> {got}" for s, n, got in wrong)
    )


@pytest.mark.skipif(
    not OPSIN_JAR.exists(), reason="OPSIN jar required for the structural invariant"
)
def test_no_emitted_name_describes_fewer_heavy_atoms_than_the_molecule():
    """No name may account for fewer heavy atoms than were drawn.

    This is the invariant the reported defect violated, asserted over the whole
    generated bridge family rather than over the four molecules that were reported.
    Abstention is permitted (the namer is allowed not to know); silently naming a
    SUBSET of the molecule is not.
    """
    _assert_family_invariant(list(_family()), min_size=40)


# A RING-bearing organyl reaches the refusal rather than the prefix builder: the
# substituent walk skips ring-bearing branches, so no owner names the branch inside
# name_polyfunctional and it must refuse the whole molecule. Refusing lets a
# downstream handler name these correctly; the `continue` that used to sit there
# dropped eight atoms apiece and shipped the bare parent. Restricted to the -OO-
# bridge, whose 'peroxy' enclosure rule is long-standing and independent of the
# shared root rule discussed in the aryl note above.
_R_GROUPS_RING = (
    "c1ccccc1",      # phenyl
    "C1CCCCC1",      # cyclohexyl
    "c1ccncc1",      # pyridinyl
)


@pytest.mark.skipif(
    not OPSIN_JAR.exists(), reason="OPSIN jar required for the structural invariant"
)
def test_ring_bearing_bridge_branch_refuses_rather_than_dropping_atoms():
    """A branch no owner can name must fail the molecule closed, not vanish.

    This is the sub-class that exercises the refusal itself: a ring-bearing
    peroxy branch is skipped by the substituent walk, so name_polyfunctional has
    to refuse. With the refusal in place these are named correctly by a
    downstream handler ('2-(phenylperoxy)ethan-1-ol'); with a bare `continue`
    they came back as 'ethan-1-ol' and 'propanoic acid', eight atoms short.
    """
    family = [
        (s, c, h)
        for s, c, h in _family(_R_GROUPS_RING)
        if "OO" in s  # deterministic peroxy morphology
    ]
    _assert_family_invariant(family, min_size=6)


@pytest.mark.skipif(
    not OPSIN_JAR.exists(), reason="OPSIN jar required for the injectivity invariant"
)
def test_distinct_molecules_never_share_one_name():
    """Two different molecules must not receive the same name.

    'CC(C)(C)OOCCO' and 'CC(C)OOCCO' both came back as 'ethan-1-ol'; a name that
    is not injective over distinct structures cannot be a structure descriptor.
    Abstentions are excluded -- many molecules may legitimately share 'unknown'.
    """
    by_name = {}
    for smiles, canonical, _heavy in _family():
        name = _name(smiles)
        if _is_abstention(name):
            continue
        by_name.setdefault(name.strip().lower(), set()).add(canonical)

    collisions = {n: s for n, s in by_name.items() if len(s) > 1}
    assert not collisions, (
        "one name for several distinct molecules:\n"
        + "\n".join(f"  {n!r} <- {sorted(s)}" for n, s in collisions.items())
    )


# ---------------------------------------------------------------------------
# The Blue-Book-cited PIN strings. These literals are legitimate: each is quoted
# from, or built by the verbatim construction of, a cited (PIN) example.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles,expected,cite",
    [
        # P-63.3.1(1) + the bare 'tert-butyl' of BB 23389.
        ("CC(C)(C)OOCCO", "2-(tert-butylperoxy)ethan-1-ol", "P-63.3.1 / BB 23389"),
        # P-57.1.3 bars 'isopropyl'; '(propan-2-yl)' keeps its own marks, which
        # forces square brackets outside -- BB 27876 '1-[(propan-2-yl)diselanyl]propane'.
        ("CC(C)OOCCO", "2-[(propan-2-yl)peroxy]ethan-1-ol", "P-63.3.1 / BB 27876"),
        # P-71.3 (BB 40737) spells 'tert-butyldisulfanyl' verbatim.
        ("CC(C)(C)SSCCO", "2-(tert-butyldisulfanyl)ethan-1-ol", "P-71.3 / BB 40737"),
        # P-63.3.2 (BB 27914) template '(methoxysulfanyl)cyclohexane', with
        # 'tert-butoxy' licensed bare in a concatenated PIN prefix by P-71.2.1.2.
        ("CC(C)(C)OSCCO", "2-(tert-butoxysulfanyl)ethan-1-ol", "P-63.3.2 / P-71.2.1.2"),
        # The exact BB 27914 morphology, with methyl.
        ("COSCCO", "2-(methoxysulfanyl)ethan-1-ol", "P-63.3.2 / BB 27914"),
        # The OTHER direction of the mixed bridge is attached through O, so it
        # ends in 'oxy' -- BB 27914 '[(methylsulfanyl)oxy]ethane'.
        ("CCSOCCO", "2-[(ethylsulfanyl)oxy]ethan-1-ol", "P-63.3.2 / BB 27914"),
    ],
)
def test_bluebook_cited_pin_strings(smiles, expected, cite):
    assert _name(smiles) == expected, f"PIN per {cite}"


def test_disulfanediyl_is_never_emitted_for_a_monovalent_branch():
    """'disulfanediyl' is P-63.3.1 method (3): the MULTIPLICATIVE divalent bridge
    -SS-. It is not a monovalent substituent prefix, so no branch disulfide may
    cite it. Verified across the whole family rather than on one molecule."""
    offenders = []
    for smiles, _canonical, _heavy in _family():
        name = _name(smiles)
        if not _is_abstention(name) and "disulfanediyl" in name:
            offenders.append((smiles, name))
    assert not offenders, (
        "divalent 'disulfanediyl' cited for a monovalent branch:\n"
        + "\n".join(f"  {s}: {n!r}" for s, n in offenders)
    )


def test_tert_butoxy_is_never_spelled_tert_butyloxy():
    """P-63.2.2.2 (BB 27679): (CH3)3C-O- is 'tert-butoxy', explicitly NOT
    'tert-butyloxy'. Guards the ether-ownership handoff that a too-broad
    fail-closed conversion broke once already."""
    for smiles in ("CC(C)(C)OCCO", "CC(C)(C)OCCC(=O)O"):
        name = _name(smiles)
        if _is_abstention(name):
            continue
        assert "butyloxy" not in name, f"{smiles} -> {name!r}"
        assert "tert-butoxy" in name, f"{smiles} -> {name!r}"


# ---------------------------------------------------------------------------
# The new primitives, in isolation. No OPSIN, no molecule pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles,frag,chalcogen,expected",
    [
        # -O-R: the CONTRACTED P-63.2.2.2 morphology.
        ("COSCCO", [0, 1], 1, "methoxy"),
        ("CCOSCCO", [0, 1, 2], 2, "ethoxy"),
        ("CCCOSCCO", [0, 1, 2, 3], 3, "propoxy"),
        ("CC(C)(C)OSCCO", [0, 1, 2, 3, 4], 4, "tert-butoxy"),
        # A locant-bearing free valence keeps the alkyl whole inside marks.
        ("CC(C)OSCCO", [0, 1, 2, 3], 3, "(propan-2-yl)oxy"),
        # -S-R / -Se-R: the organyl is cited then the chalcogen free valence.
        ("CSOCCO", [0, 1], 1, "methylsulfanyl"),
        ("CC[Se]OCCO", [0, 1, 2], 2, "ethylselanyl"),
    ],
)
def test_composed_chalcogen_group_prefix(smiles, frag, chalcogen, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert composed_chalcogen_group_prefix(mol, frag, chalcogen, set()) == expected


def test_composed_chalcogen_group_prefix_defers_on_a_second_chalcogen():
    """A di-chalcogen continuation is the P-63.3.1 peroxy/disulfanyl class, which
    the shared cascade already names whole. The primitive must decline there so
    that working path is never displaced."""
    mol = Chem.MolFromSmiles("COOCCO")  # C0 O1 O2 C3 C4 O5
    assert composed_chalcogen_group_prefix(mol, [0, 1, 2], 1, set()) is None


def test_composed_chalcogen_group_prefix_fails_closed():
    """Non-chalcogen hub, a chalcogen with no organyl continuation, and a hub
    outside the fragment must all refuse rather than guess a morphology."""
    mol = Chem.MolFromSmiles("CCO")  # C0 C1 O2
    # A carbon hub is not a chalcogen at all.
    assert composed_chalcogen_group_prefix(mol, [0, 1, 2], 0, set()) is None
    # Hub outside the fragment.
    assert composed_chalcogen_group_prefix(mol, [0, 1], 2, set()) is None
    # No organyl continuation INSIDE the fragment: the O is the whole fragment,
    # i.e. a terminal -OH, which is a suffix/hydroxy question and not an -O-R
    # prefix. (With C1 in the fragment the same O legitimately names 'ethoxy';
    # what makes this a refusal is the absence of a continuation, not the atom.)
    assert composed_chalcogen_group_prefix(mol, [2], 2, set()) is None
    # Same shape expressed via the boundary rather than the fragment.
    assert composed_chalcogen_group_prefix(mol, [0, 1, 2], 2, {1}) is None
    # A BRANCHED hub (two in-fragment continuations) is not the simple -X-R
    # shape and must refuse rather than pick one arbitrarily.
    diether = Chem.MolFromSmiles("COC")  # C0 O1 C2
    assert composed_chalcogen_group_prefix(diether, [0, 1, 2], 1, set()) is None


@pytest.mark.parametrize(
    "smiles,attach,frag,expected_any,expected_mixed",
    [
        # -O-O- homo bridge: a dichalcogen bridge, but NOT a mixed one.
        ("COOCCO", 1, {1, 2, 0}, True, False),
        # -S-S- homo bridge.
        ("CSSCCO", 1, {1, 2, 0}, True, False),
        # -O-S- mixed bridge: both predicates true.
        ("COSCCO", 2, {0, 1, 2}, True, True),
        # A MONO ether oxygen is neither.
        ("COCCO", 1, {0, 1}, False, False),
    ],
)
def test_is_dichalcogen_bridge_attach(
    smiles, attach, frag, expected_any, expected_mixed
):
    mol = Chem.MolFromSmiles(smiles)
    assert is_dichalcogen_bridge_attach(mol, attach, frag) is expected_any
    assert (
        is_dichalcogen_bridge_attach(mol, attach, frag, require_different=True)
        is expected_mixed
    )


@pytest.mark.parametrize(
    "smiles,frag,attach,parent,expected",
    [
        # Attached through S with an inner O -> the contracted alkoxy + 'sulfanyl',
        # enclosed per P-16.5.1.1 (BB 27914 '(methoxysulfanyl)cyclohexane').
        ("CC(C)(C)OSCCO", [0, 1, 2, 3, 4, 5], 5, {6}, "(tert-butoxysulfanyl)"),
        ("COSCCO", [0, 1, 2], 2, {3}, "(methoxysulfanyl)"),
        ("CC(C)OSCCO", [0, 1, 2, 3, 4], 4, {5}, "[(propan-2-yl)oxysulfanyl]"),
        # Attached through O with an inner S -> ends in 'oxy'. Pre-existing path,
        # asserted here so the redirect above cannot silently capture it.
        ("CCSOCCO", [0, 1, 2, 3], 3, {4}, "(ethylsulfanyl)oxy"),
        # A DI-chalcogen inner stays on the cascade: '(methylperoxy)sulfanyl'.
        ("CSOOC", [1, 2, 3, 4], 1, {0}, "(methylperoxy)sulfanyl"),
    ],
)
def test_name_mixed_chalcogen_branch(smiles, frag, attach, parent, expected):
    mol = Chem.MolFromSmiles(smiles)
    assert _name_mixed_chalcogen_branch(mol, frag, attach, parent) == expected
