"""v28 Composer1 Task 1: detach-and-name ring-substituent primitive.

Fragment-level unit test for
``orthonym.assembly.substituent_enumerator._detach_and_name_ring_substituent``
-- the reusable FIRST primitive for the always-emit recursive substituent
composer. Given a substituent fragment's ring atoms and an attachment ring
atom, it detaches the ring core and names it via the existing general
(von-Baeyer/spiro/cage) ring engine, returning a ``-yl``/``-ylidene`` token.

No ``source=='general_engine'`` provenance assertion here -- that applies to
the end-to-end tests in later v28 composer tasks. This test calls the
fragment-level primitive directly, offline.
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym


def _be():
    # best-effort / general-engine namer; OPSIN gates off for offline
    # structural assertions.
    return Orthonym(style="pin", general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True,
                     _disable_opsin_validity_gate=True,
                     _disable_grammar_validation=True)


def _be_rt():
    # same but with the OPSIN RT gate ON -- for RT-valid assertions (needs
    # Java; skip if absent).
    return Orthonym(style="pin", general_fallback=True,
                     general_fallback_unverified=True,
                     allow_aromatic_general=True)


def test_detach_and_name_isolated_cage_fragment():
    from orthonym.assembly.substituent_enumerator import _detach_and_name_ring_substituent
    # adamantane attached via a ring carbon (whole molecule = adamantan-1-yl-acetic acid)
    smi = "OC(=O)CC12CC3CC(CC(C3)C1)C2"
    mol = Chem.MolFromSmiles(smi)
    # frag = the adamantane ring atoms; attach = the ring C bonded to the CH2
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    attach = next(i for i in ring_atoms
                  if any((not mol.GetAtomWithIdx(n.GetIdx()).IsInRing())
                         for n in mol.GetAtomWithIdx(i).GetNeighbors()))
    name = _detach_and_name_ring_substituent(mol, ring_atoms, attach, allow_mancude=True)
    assert name and name != "substituent" and " " not in name
    assert name.endswith("yl") or name.endswith("ylidene")


# ============================================================================
# v28 Composer1 Task 2: recursive decoration composition (end-to-end via engine)
# ============================================================================


@pytest.mark.parametrize("smi", [
    "C[C@@H](N)c1ccc(OCc2ccc(Cl)cc2)nc1",       # pyridine core + -O-CH2-(4-Cl-phenyl) decoration
    "CC(C)(C)Sc1ccc(-c2nc3ccccc3c(=O)[nH]2)cc1", # benzene core + -S-C(CH3)3 decoration
])
def test_decorated_ring_substituent_emits_via_engine(smi):
    """A decorated ring substituent must EMIT — that is what this test protects.

    v29 Phase 6 relaxed the provenance assertion from `== "general_engine"` to
    "engine or better". The tert-butylsulfanyl case now resolves on the PIN path
    as `2-[4-(tert-butylsulfanyl)phenyl]quinazolin-4(3H)-one` (T1, round-trips to
    the input) because the decorated-CARBOCYCLIC producer added in Phase 6 can
    now supply `4-(tert-butylsulfanyl)phenyl`, letting the PIN path complete
    instead of falling through to the best-effort engine.

    That is a T4 -> T1 PROMOTION, i.e. the outcome this project wants, so
    pinning the source to `general_engine` would be pinning the weaker result.
    The first parameter still exercises the engine path (T3), so engine coverage
    is not lost.
    """
    row = _be().name_tiered(smi)
    assert row["source"] in ("general_engine", "pin_path"), row
    assert row["name"] and "unknown" not in row["name"] and " substituent" not in row["name"]


# ============================================================================
# v28 Composer1 Task 2b: polycyclic (fused / bridged) decorated-core composition
# ============================================================================


@pytest.mark.parametrize("smi,expected", [
    # FUSED decorated core
    ("OC(=O)Cc1ccc2cc(Cl)ccc2c1", "(6-chloronaphthalen-2-yl)acetic acid"),
    # BRIDGED decorated core
    ("OC(=O)CC12CC3CC(O)(CC(C3)C1)C2", "(3-hydroxyadamantan-1-yl)acetic acid"),
])
def test_polycyclic_decorated_substituent_is_named(smi, expected):
    """A decorated polycyclic ring substituent must be NAMED, and named
    correctly.

    v30 P3-T1c: this test used to assert ``row["source"] == "general_engine"``.
    That pinned an internal ROUTE, and the route changed when the composer's
    ring-substituent path gained access to the general tier -- these molecules are
    now named earlier, by the composer. The PROVENANCE NOTE at :101-104 of this
    file already documented route assertions as fragile for exactly this reason,
    so the assertion is now on the NAME, which is the property that matters.

    Both expected names are the ones this test's own parametrisation comments
    named from the start, and neither route ever produced the second one before:

    * ``acetic acid`` is the PIN, not ``ethanoic acid``. **P-21.1.1 / Table 28.1
      context, stated at ``BlueBookV2/BlueBookV2.md:2004``**: *"A special class of
      parent structures having retained names ... is called functional parent
      compounds, for example, phenol and acetic acid. These two names are
      preferred IUPAC names; the corresponding systematic alternatives, benzenol
      and ethanoic acid, may be used in general IUPAC nomenclature."*
    * ``adamantan-1-yl`` is the PIN stem, not ``tricyclo[3.3.1.1^3,7]decan-3-yl``.
      **P-23.7 "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES"** (``:9879``):
      *"The retained names adamantane and cubane are used in general nomenclature
      and as preferred IUPAC names."* Table 2.6 (``:9885``) prints *"adamantane
      (PIN) tricyclo[3.3.1.1^3,7]decane"* -- retained name PIN, descriptor the
      alternative.
    * Both names round-trip through OPSIN 2.9.0 to the input's FULL InChIKey
      (not skeleton-only): verified this session.
    """
    row = _be().name_tiered(smi)
    assert row["name"] == expected, row
    assert "unknown" not in row["name"] and " substituent" not in row["name"]


# ============================================================================
# v28 Composer1 Task 2c: fused-HETEROCYCLE decorated-core composition
# ============================================================================
#
# PROVENANCE NOTE. The whole-molecule PIN path ALREADY names decorated
# fused-heterocycle substituents (its fused-ring substituent machinery), so for
# a PIN-nameable molecule ``name_tiered`` reports ``source=='pin_path'`` and the
# general-engine composer is never the winning tier offline. The new
# general-engine fused-heterocycle branch is therefore exercised DIRECTLY at the
# composer here -- ``_recursive_fragment_substituent_name`` returns ``None``
# unless ``allow_mancude=True`` (its docstring: reached from ``name_substituent``
# ONLY under the general-fallback / general-engine context), so this IS the
# general-engine code path and nothing else, mirroring the Task 1 primitive test
# above. (End-to-end value verified out-of-band with ``diagnose.py --complete``:
# for complex decorations the PIN path silently drops -- e.g.
# ``OC(=O)Cc1c[nH]c2ccc(OCc3ccc(Cl)cc3)cc12`` -- the PIN name fails the SELF-01
# round-trip in production and the general engine recovers the full retained
# name ``2-(5-[(4-chlorophenyl)methoxy]-1H-indol-3-yl)ethanoic acid``.)


def _ring_substituent_fragment(mol, chain_atom):
    """Ring-substituent fragment = the ring system + its own decorations, with
    the parent-chain side (``chain_atom``) excluded; return ``(frag_atoms,
    attach)`` where ``attach`` is the ring atom bonded to ``chain_atom``."""
    ring = {a for r in mol.GetRingInfo().AtomRings() for a in r}
    attach = next(i for i in ring
                  if mol.GetBondBetweenAtoms(i, chain_atom) is not None)
    frag = set(ring)
    stack = list(ring)
    while stack:
        a = stack.pop()
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            ni = nb.GetIdx()
            if ni in frag or ni == chain_atom or nb.GetAtomicNum() <= 1:
                continue
            frag.add(ni)
            stack.append(ni)
    return sorted(frag), attach


@pytest.mark.parametrize("smi,chain_atom,expected", [
    # (5-chloro-1H-indol-3-yl)acetic acid — fused N-heterocycle core + Cl deco
    ("OC(=O)Cc1c[nH]c2ccc(Cl)cc12", 3, "5-chloro-1H-indol-3-yl"),
    # (2-methylquinolin-6-yl)acetic acid — fused N-heterocycle core + Me deco
    ("OC(=O)Cc1ccc2nc(C)ccc2c1", 3, "2-methylquinolin-6-yl"),
])
def test_fused_heterocycle_decorated_substituent_composes_via_engine(
        smi, chain_atom, expected):
    from orthonym.assembly.substituent_enumerator import (
        _recursive_fragment_substituent_name)
    mol = Chem.MolFromSmiles(smi)
    frag, attach = _ring_substituent_fragment(mol, chain_atom)
    # PIN default (allow_mancude=False) MUST decline -> byte-identity guard.
    assert _recursive_fragment_substituent_name(
        mol, frag, attach, allow_mancude=False) is None
    # general-engine path (allow_mancude=True) composes the decorated retained
    # fused-heterocycle -yl token.
    name = _recursive_fragment_substituent_name(
        mol, frag, attach, allow_mancude=True)
    assert name == expected, name
    assert " " not in name and "unknown" not in name.lower()


@pytest.mark.xfail(strict=False, reason=(
    "PIN-path pre-emption: the PIN fused-ring substituent machinery already "
    "names these (RT-valid), so name_tiered reports source=='pin_path' offline. "
    "The general-engine fused-heterocycle composer is the winning source only in "
    "production when the PIN name fails SELF-01 (complex dropped decorations). "
    "See the direct-composer test above for the general-engine path proof."))
@pytest.mark.parametrize("smi", [
    "OC(=O)Cc1c[nH]c2ccc(Cl)cc12",     # (5-chloro-1H-indol-3-yl)acetic acid
    "OC(=O)Cc1ccc2nc(C)ccc2c1",        # (2-methylquinolin-6-yl)acetic acid
])
def test_fused_heterocycle_decorated_substituent_emits_via_engine(smi):
    row = _be().name_tiered(smi)
    assert row["source"] == "general_engine", row
    assert row["name"] and "unknown" not in row["name"] and " substituent" not in row["name"]


# ============================================================================
# v28 Composer1 Task 3: heteroatom carrier in the RING-ON-CHAIN branch
# ============================================================================
#
# PROVENANCE NOTE (mirrors Task 2c above). This T0 case-4 molecule's PIN path
# ALREADY fails to name the -CH2-S-(decorated fused-heterocycle) substituent
# (the all-carbon carrier guard in ``_compound_ring_on_chain_substituent``
# rejects the sulfur linker), so whether ``name_tiered``'s whole-molecule
# ``source`` comes back ``'general_engine'`` depends on whether the PIN path
# happens to name the REST of the molecule cleanly through some other route
# too. Primary correctness assertion is the direct function-level test below
# (calls ``_compound_ring_on_chain_substituent`` on the isolated
# ``-CH2-S-(ring)`` fragment); the whole-molecule provenance test is kept but
# marked xfail(strict=False) if the PIN path pre-empts it.


@pytest.mark.xfail(strict=False, reason=(
    "PIN-path pre-emption risk (mirrors Task 2c): whole-molecule provenance "
    "depends on the PIN path failing SELF-01 for this exact molecule, which "
    "is not guaranteed offline. Primary correctness proof is the direct "
    "function-level test below."))
def test_ring_on_chain_heteroatom_carrier_emits_via_engine():
    smi = "Cc1cccn2cc(CSc3nc4scc(C5CC5)c4c(=O)n3-c3ccccc3)nc12"
    row = _be().name_tiered(smi)
    assert row["source"] == "general_engine", row
    assert row["name"] and "unknown" not in row["name"]


def test_ring_on_chain_sulfanyl_carrier_direct_composer():
    """Direct function-level test (primary correctness assertion): the
    isolated -CH2-S-(decorated fused-heterocycle) substituent fragment from
    the T0 case-4 molecule, named directly via
    ``_compound_ring_on_chain_substituent``. PIN default (allow_mancude=False)
    MUST decline -> byte-identity guard; allow_mancude=True must emit a real
    linker name (never None, never the 'substituent' sentinel, never a
    space-bearing token) that carries BOTH the sulfanyl connective and the
    recursively-composed ring-yl core."""
    from orthonym.rules.ring_substituents import (
        _compound_ring_on_chain_substituent)
    smi = "Cc1cccn2cc(CSc3nc4scc(C5CC5)c4c(=O)n3-c3ccccc3)nc12"
    mol = Chem.MolFromSmiles(smi)
    # atom 8 = the CH2 carrier carbon (parent-side attach is atom 7); atom 9 =
    # the carrier sulfur; atoms 10-28 = the decorated thieno-pyrimidinone core
    # (thiophene ring + pyrimidinone ring + cyclopropyl + oxo + N-phenyl).
    attach_idx = 8
    frag_atoms = list(range(8, 29))
    frag_set = set(frag_atoms)
    ring_info = mol.GetRingInfo()
    frag_ring_atoms = {a for a in frag_atoms if ring_info.NumAtomRings(a) > 0}

    # PIN default MUST decline (byte-identity guard): the sulfur carrier is
    # rejected exactly as before this task's change.
    assert _compound_ring_on_chain_substituent(
        mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info,
        allow_mancude=False,
    ) is None

    name = _compound_ring_on_chain_substituent(
        mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info,
        allow_mancude=True,
    )
    assert name is not None and name != "substituent", name
    assert " " not in name
    assert "sulfanyl" in name
    assert name.endswith("methyl")


def test_ring_on_chain_amino_carrier_direct_composer():
    """Mirrors ``test_ring_on_chain_sulfanyl_carrier_direct_composer`` above
    but for the N (amino) connective: an ordinary ``-CH2-NH-(ring)`` secondary
    amine carrier. Reviewer finding (T3 fix): a neutral, non-aromatic,
    degree-2 bridging N ALWAYS carries exactly 1 implicit H (unlike O/S,
    which carry 0), so the original ``GetTotalNumHs() == 0`` guard applied
    uniformly to S/O/N made the 'amino' connective branch permanently
    unreachable dead code. PIN default (allow_mancude=False) MUST decline ->
    byte-identity guard; allow_mancude=True must emit a real linker name
    (never None, never the 'substituent' sentinel, never a space-bearing
    token) that carries BOTH the amino connective and the recursively-composed
    ring-yl core."""
    from orthonym.rules.ring_substituents import (
        _compound_ring_on_chain_substituent)
    # 2-[(carrier)]butanoic acid parent with a -CH2-NH-(5-chloro-1H-indol-3-yl)
    # carrier (an N analog of the shipped -CH2-S- case above).
    smi = "CCC(CNc1c[nH]c2ccc(Cl)cc12)C(=O)O"
    mol = Chem.MolFromSmiles(smi)
    # atom 3 = the CH2 carrier carbon (parent-side attach is atom 2); atom 4 =
    # the carrier nitrogen; atoms 5-14 = the decorated 5-chloroindol-3-yl core.
    attach_idx = 3
    frag_atoms = list(range(3, 15))
    frag_set = set(frag_atoms)
    ring_info = mol.GetRingInfo()
    frag_ring_atoms = {a for a in frag_atoms if ring_info.NumAtomRings(a) > 0}

    # PIN default MUST decline (byte-identity guard): the nitrogen carrier is
    # rejected exactly as before this fix.
    assert _compound_ring_on_chain_substituent(
        mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info,
        allow_mancude=False,
    ) is None

    name = _compound_ring_on_chain_substituent(
        mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info,
        allow_mancude=True,
    )
    assert name is not None and name != "substituent", name
    assert " " not in name
    assert "amino" in name
    assert name.endswith("methyl")


# ============================================================================
# v28 Composer1 Task 4: wire the composer into name_substituent + de-mask
# ============================================================================
#
# The core wiring (Task 2, commit bcbb6745) already routes a ring-bearing
# fragment declined by every narrow namer through
# ``_recursive_fragment_substituent_name`` under ``allow_mancude=True``, and
# de-masks the ``'substituent'`` sentinel to a clean ``None`` when the
# composer also declines. This is the always-emit contract test: the general
# path must never leak the raw sentinel (or any space-bearing token) past
# ``name_substituent``.


def test_general_substituent_never_returns_sentinel_when_decomposable():
    """adamantyl-acetic-acid: the adamantane ring-substituent fragment is
    declined by every PIN-tier namer (PIN-only ``name_tiered`` abstains
    entirely -- 'unknown organic compound', source=='abstain' -- verified
    offline) but the general-fallback recursive composer names it as a real
    von-Baeyer cage -yl token. Primary assertion is the direct call (no
    OPSIN needed); the end-to-end provenance assertion also holds here
    because this molecule's PIN path genuinely fails closed rather than
    pre-empting the general engine (verified out-of-band: PIN-only
    ``name_tiered`` returns source=='abstain', not 'pin_path', and the
    composed name round-trips via ``diagnose.py --complete``:
    '2-(tricyclo[3.3.1.1^3,7]decan-3-yl)ethanoic acid')."""
    from orthonym.assembly.substituent_enumerator import name_substituent
    smi = "OC(=O)CC12CC3CC(CC(C3)C1)C2"  # adamantyl-acetic-acid
    mol = Chem.MolFromSmiles(smi)
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    attach = next(i for i in ring_atoms
                  if any((not mol.GetAtomWithIdx(n.GetIdx()).IsInRing())
                         for n in mol.GetAtomWithIdx(i).GetNeighbors()))

    # PIN default (allow_mancude=False) MUST keep the exact pre-existing
    # sentinel behavior -> byte-identity guard.
    pin_name = name_substituent(mol, ring_atoms, attach, allow_mancude=False)
    assert pin_name == "substituent"

    # General path: a real name or a clean None -- NEVER the sentinel, NEVER
    # a space-bearing token.
    name = name_substituent(mol, ring_atoms, attach, allow_mancude=True)
    assert name is None or (name != "substituent" and " " not in name)
    # v30 P3-T1c: was ``tricyclo[3.3.1.1^3,7]decan-3-yl``. The retained name is
    # the PIN -- **P-23.7 "RETAINED NAMES FOR VON BAEYER PARENT HYDRIDES"**
    # (``BlueBookV2/BlueBookV2.md:9879``): *"The retained names adamantane and
    # cubane are used in general nomenclature and as preferred IUPAC names."*
    # Table 2.6 (``:9885``) prints *"adamantane (PIN) tricyclo[3.3.1.1^3,7]
    # decane"*, i.e. the descriptor is the ALTERNATIVE. The locant also drops
    # 3 -> 1 because the free valence now takes the lowest locant (P-29.3.2);
    # OPSIN resolves ``adamantan-N-ol`` and ``tricyclo[3.3.1.1^3,7]decan-N-ol``
    # to the same InChIKey for all N in 1..10, so the two numberings coincide
    # and the stem swap is locant-safe.
    assert name == "adamantan-1-yl", name

    # End-to-end: assert the NAME, not the route. This used to require
    # ``source == "general_engine"``; the composer now names it first (P3-T1c
    # gave its ring path access to the general tier), so the route is no longer a
    # stable property -- the name is. Round-trips to the input's FULL InChIKey.
    assert _be().name_tiered(smi)["name"] == "(adamantan-1-yl)acetic acid"


# ============================================================================
# v28 Composer1 Task 5: discover_substituents partition robustness
# (substituents off suffix/FG atoms)
# ============================================================================
#
# T0 case-5 molecule: an N-aryl amide anilide (the 3-fluorophenyl hangs off the
# amide nitrogen -- a SUFFIX/FG atom, unreachable from the parent-chain walk) PLUS
# a piperidine-borne fused-heterocycle ring that no honest namer can express. The
# general-engine chain path partitions parent = chain | suffix and the N-aryl ring
# atoms land unassigned. Under the general-fallback context this must FAIL CLOSED
# to a clean abstain (never a wrong name that drops the fluorophenyl, never a hard
# crash) -- the pipeline must NOT raise; either it names the whole molecule via the
# engine (unreachable here -- the piperidine substituent is unnameable, multi-
# blocked on Composer #2) OR it abstains cleanly (no name surfaced).


def test_substituent_off_amide_nitrogen_is_partitioned():
    smi = "CC(C)(C(=O)Nc1cccc(F)c1)N1CCC(c2nc(-c3cc4ccccc4o3)cs2)CC1"
    row = _be().name_tiered(smi)     # must NOT raise
    assert row is not None
    # v30 P3-T1c: was ``(row["source"] == "general_engine") or (not
    # row.get("name"))`` -- a route assertion with an abstain escape hatch. The
    # molecule is no longer multi-blocked (the composer's ring path reaches the
    # general tier now), so it NAMES, and the property this test exists to
    # protect is stated directly instead: the N-aryl ring that hangs off the
    # amide nitrogen must still be in the name. Dropping the fluorophenyl was
    # the wrong-structure failure the test was written against.
    assert row["name"], row
    assert "N-(3-fluorophenyl)" in row["name"], row
    assert "unknown" not in row["name"] and " substituent" not in row["name"]
    assert row["name"] == (
        "2-({4-[4-(1-benzofuran-2-yl)-1,3-thiazol-2-yl]piperidin-1-yl})"
        "-2-methyl-N-(3-fluorophenyl)propanamide"), row


# ============================================================================
# v28 Composer1 Task C1-T6: assembly-robustness instrument smoke test
# ============================================================================
#
# TINY shape-only smoke test for `` -- the
# offline, hang-safe per-fragment naming-success measurement instrument (NOT
# gated production; see the module docstring there for the decomposition
# method + success criterion). This test only asserts the returned dict's
# SHAPE (keys present, n correct, rates in [0, 1]) -- it does NOT assert any
# specific rate, since the proxy parent-selection heuristic is deliberately
# simplified (see that module's docstring) and the exact numbers are not a
# contract of this task.


def test_asm_robustness_instrument_smoke():
    import importlib.util
    from pathlib import Path

    import sys

    project_root = Path(__file__).parent.parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "asm_robustness", str(project_root / "" / "asm_robustness.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass decorator needs the module registered
    spec.loader.exec_module(mod)

    # 5 SMILES already used above (T2 / T2b / T4 cases).
    smiles = [
        "C[C@@H](N)c1ccc(OCc2ccc(Cl)cc2)nc1",       # T2
        "CC(C)(C)Sc1ccc(-c2nc3ccccc3c(=O)[nH]2)cc1", # T2
        "OC(=O)Cc1ccc2cc(Cl)ccc2c1",                 # T2b
        "OC(=O)CC12CC3CC(O)(CC(C3)C1)C2",            # T2b
        "OC(=O)CC12CC3CC(CC(C3)C1)C2",                # T4 (adamantyl-acetic acid)
    ]

    result = mod.measure_assembly_robustness(smiles)

    assert set(result.keys()) == {"fragment_p", "mol_all_named", "n"}
    assert result["n"] == 5
    assert 0.0 <= result["fragment_p"] <= 1.0
    assert 0.0 <= result["mol_all_named"] <= 1.0
