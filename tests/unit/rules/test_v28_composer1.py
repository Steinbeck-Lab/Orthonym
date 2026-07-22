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
    row = _be().name_tiered(smi)
    assert row["source"] == "general_engine", row
    assert row["name"] and "unknown" not in row["name"] and " substituent" not in row["name"]


# ============================================================================
# v28 Composer1 Task 2b: polycyclic (fused / bridged) decorated-core composition
# ============================================================================


@pytest.mark.parametrize("smi", [
    "OC(=O)Cc1ccc2cc(Cl)ccc2c1",              # (6-chloronaphthalen-2-yl)acetic acid — FUSED decorated core
    "OC(=O)CC12CC3CC(O)(CC(C3)C1)C2",         # (3-hydroxyadamantan-1-yl)acetic acid — BRIDGED decorated core
])
def test_polycyclic_decorated_substituent_emits_via_engine(smi):
    row = _be().name_tiered(smi)
    assert row["source"] == "general_engine", row
    assert row["name"] and "unknown" not in row["name"] and " substituent" not in row["name"]


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
    assert name == "tricyclo[3.3.1.1^3,7]decan-3-yl", name

    # End-to-end: the general engine (not the PIN path) wins this molecule.
    assert _be().name_tiered(smi)["source"] == "general_engine"
