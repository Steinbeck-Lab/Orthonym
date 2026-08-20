"""v33 glyco composer, slice 1 — NON-REDUCING oligosaccharides (3+ units).

Raffinose is the canonical witness: a non-reducing trisaccharide
(alpha-D-Gal-(1->6)-alpha-D-Glc central-linked (1<->2) to beta-D-Fru) that the
reducing-chain assembler (`_oligo_topology`) and the binary assembler
(`_count_sugar_rings>=3`) both fail closed on, so it currently emits `unknown`.

The expected name is OPSIN-RT-verified to raffinose's InChIKey (MUPFEKGTMRGPLJ).
"""
from rdkit import Chem
from orthonym.rules import oligosaccharides as O

RAFFINOSE = "OC[C@H]1O[C@H](OC[C@H]2O[C@H](O[C@]3(CO)O[C@H](CO)[C@@H](O)[C@@H]3O)[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@H]1O"
EXPECTED = "alpha-D-galactopyranosyl-(1->6)-alpha-D-glucopyranosyl beta-D-fructofuranoside"


def test_raffinose_nonreducing_trisaccharide():
    mol = Chem.MolFromSmiles(RAFFINOSE)
    assert mol is not None
    name = O.name_nonreducing_oligosaccharide(mol)
    assert name == EXPECTED, f"got {name!r}"


def test_raffinose_via_public_entry():
    # name_disaccharide (the public P-102.7 entry) must now route raffinose too
    mol = Chem.MolFromSmiles(RAFFINOSE)
    assert O.name_disaccharide(mol) == EXPECTED


# --- slice 2: BRANCHED reducing oligosaccharides (P-102.7.3) ---
# a unit accepting >1 glycosyl; both expected names are OPSIN-RT-verified to the input.
BRANCHED_GLUCOTRIOSE = "OC[C@H]1O[C@H](OC[C@H]2OC(O)[C@H](O)[C@@H](O)[C@@H]2O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"
BRANCHED_GLUCOTRIOSE_NAME = "alpha-D-glucopyranosyl-(1->6)-[alpha-D-glucopyranosyl-(1->4)]-D-glucopyranose"


def test_branched_glucotriose():
    mol = Chem.MolFromSmiles(BRANCHED_GLUCOTRIOSE)
    assert mol is not None
    assert O.name_branched_oligosaccharide(mol) == BRANCHED_GLUCOTRIOSE_NAME


def test_branched_via_public_entry():
    mol = Chem.MolFromSmiles(BRANCHED_GLUCOTRIOSE)
    assert O.name_disaccharide(mol) == BRANCHED_GLUCOTRIOSE_NAME


def test_branched_lewis_type():
    # beta-D-Gal-(1->3)-[alpha-L-Fuc-(1->4)]-D-Glc (a Lewis-a core) round-trips.
    smi = "C[C@@H]1O[C@@H](O[C@H]2[C@H](O[C@@H]3O[C@H](CO)[C@H](O)[C@H](O)[C@H]3O)[C@@H](O)C(O)O[C@@H]2CO)[C@@H](O)[C@H](O)[C@@H]1O"
    mol = Chem.MolFromSmiles(smi)
    name = O.name_branched_oligosaccharide(mol)
    assert name == "alpha-L-fucopyranosyl-(1->4)-[beta-D-galactopyranosyl-(1->3)]-D-glucopyranose", name


def test_branched_declines_linear():
    # a LINEAR reducing chain (no branch point) must fall through to the linear
    # namer, not the branched one.
    maltose = "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O"
    m = Chem.MolFromSmiles(maltose)
    assert m is not None
    assert O.name_branched_oligosaccharide(m) is None


def test_dispatch_precondition_routes_extended_oligo():
    # DISPATCH INTEGRATION (closes the choke-point-off-path blind spot): the cheap
    # no-OPSIN precondition must fire for non-reducing AND branched oligosaccharides,
    # else name_tiered routes them to the general oxane engine and the composer,
    # though correct when called directly, is never reached.
    assert O._has_extended_oligo(Chem.MolFromSmiles(RAFFINOSE)) is True
    assert O._has_extended_oligo(Chem.MolFromSmiles(BRANCHED_GLUCOTRIOSE)) is True
    # a single monosaccharide / disaccharide must NOT trip it.
    assert O._has_extended_oligo(Chem.MolFromSmiles("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")) is False


def test_nonreducing_declines_under_three_units():
    # the new 3+ namer must fail closed on <3 sugar units (a monosaccharide /
    # disaccharide is the single-sugar / binary assembler's job — no double-handling).
    glucose = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"  # 1 unit
    m = Chem.MolFromSmiles(glucose)
    assert m is not None
    assert O.name_nonreducing_oligosaccharide(m) is None


# --- v33 Task 1.3 (breadth-lever program, Phase 1) — glycan SCALE ---
#
# SPY finding (measured,  family
# `sugar_glycan`, filtered to real >=3-linked-sugar-unit trees via this
# module's own `_detect_sugar_units_links`): the single dominant, well-defined
# scale cap on real >=3-ring backlog witnesses is that `name_branched_
# oligosaccharide`'s reducing-terminus detection REQUIRED a free hemiacetal
# -OH (`GetTotalNumHs() > 0`) with no relaxation at all -- `_oligo_topology`
# (used only by the LINEAR namer) already had an alkyl/aryl-capped-terminus
# relaxation (v33 Engine-2 fix (b)), but it was never generalized to the
# BRANCHED namer, so ANY branched tree sitting behind a capped root (the
# common synthetic aminoethyl/aminopentyl glycoconjugate linker included) was
# structurally unreachable regardless of how many residues it had -- 37/85
# still-abstaining real oligo>=3 witnesses hit exactly this decline site
# (`if len(reducing) != 1: return None`).
#
# A second, independent bug found by the same SPY: `_glycoside_cap_name`
# always guessed `attach_locant=1` when converting the isolated cap fragment's
# own free-molecule name into a substituent prefix (P-29.2) -- correct only
# when the free valence happens to sit at the fragment's own C1, and silently
# WRONG (declining a nameable cap) whenever a senior group claims that locant
# instead, e.g. `pentan-1-amine` for a 5-aminopentyl linker (amine at C1,
# attachment at C5). Root-cause fix: try `_located_acyclic_alkyl_name`
# (`assembly/substituent_naming.py`) FIRST -- it derives the free-valence
# locant FROM THE STRUCTURE (P-46.1.8) rather than guessing -- falling back to
# the old name+convert path only for what it declines (aromatic/ring caps).
#
# Fixed together: `name_branched_oligosaccharide` now detects a capped root
# exactly like `_oligo_topology` does, and renders it via the same
# P-102.5.6.2.2 "{cap} n-O-{substituent}-{glycosideHead}" form the linear
# namer already uses, generalized so the O-substituent can be a full
# recursively-rendered branched subtree (`_render`), not just a linear chain.
#
# Measured over the FULL 413-row `sugar_glycan` backlog family (0 regressions,
# 0 new exceptions): exactly 4 NEW conversions (idx 100, 264, 375, 376 in that
# file) -- all full-InChIKey OPSIN round-trip verified. Three are witnessed
# here directly by SMILES (not by census index, which drifts as prior tasks
# land); each was confirmed to return ``None`` from `name_branched_
# oligosaccharide` against the pre-Task-1.3 code (git HEAD `3bd5d3b1`) before
# this fix, and a real name after it.

GIANT_AMINOETHYL_TRIDECASACCHARIDE = (
    "NCCO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H]"
    "(O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@@H]4O[C@H](CO)[C@@H](O)[C@H]"
    "(O[C@@H]5O[C@H](CO[C@@H]6O[C@H](CO[C@@H]7O[C@H](CO[C@@H]8O[C@H]"
    "(CO[C@@H]9O[C@H](CO)[C@@H](O)[C@H](O)[C@H]9O)[C@@H](O)[C@H](O)[C@H]8O)"
    "[C@@H](O)[C@H](O)[C@H]7O)[C@@H](O)[C@H](O)[C@H]6O)[C@@H](O)[C@H]"
    "(O[C@@H]6O[C@H](CO)[C@@H](O)[C@H](O[C@@H]7O[C@H](CO)[C@@H](O)[C@H]"
    "(O[C@@H]8O[C@H](CO)[C@@H](O)[C@H](O[C@@H]9O[C@H](CO)[C@@H](O)[C@H](O)"
    "[C@H]9O)[C@H]8O)[C@H]7O)[C@H]6O)[C@H]5O)[C@H]4O)[C@H]3O)[C@H]2O)[C@H]1O"
)

AMINOPENTYL_URONIC_BRANCHED_OCTASACCHARIDE = (
    "NCCCCCO[C@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O[C@@H]3O[C@H]"
    "(CO)[C@@H](O)[C@H](O[C@H]4O[C@H](CO)[C@@H](O[C@@H]5O[C@H](CO)[C@@H]"
    "(O[C@@H]6O[C@H](CO)[C@@H](O)[C@H](O)[C@H]6O)[C@H](O[C@H]6O[C@H](C(=O)O)"
    "[C@@H](O)[C@H](O)[C@H]6O)[C@@H]5O)[C@H](O)[C@H]4O)[C@H]3O)[C@H]"
    "(O[C@H]3O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]3O)[C@@H]2O)[C@H](O)[C@H]1O"
)

METHYL_ESTER_HEXYL_HEXASACCHARIDE = (
    "COC(=O)CCCCCO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@@H]"
    "(O)[C@H](O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@@H]4O[C@H](CO)[C@@H]"
    "(O)[C@H](O[C@@H]5O[C@H](CO)[C@@H](O)[C@H](O[C@@H]6O[C@H](CO)[C@@H]"
    "(O)[C@H](O)[C@H]6O)[C@H]5O)[C@H]4O)[C@H]3O)[C@H]2O)[C@H]1O"
)


def _full_inchikey(smiles):
    from rdkit.Chem import inchi
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


def test_branched_with_capped_aminoethyl_root_giant_tree():
    # v33 Task 1.3 SCALE witness: a 13-unit branched glucan tree behind a
    # simple aminoethyl-capped root -- unreachable before this fix regardless
    # of size (name_branched_oligosaccharide required a FREE reducing -OH).
    mol = Chem.MolFromSmiles(GIANT_AMINOETHYL_TRIDECASACCHARIDE)
    assert mol is not None
    name = O.name_disaccharide(mol)
    assert name is not None, "expected a name, got an abstention"
    assert name.startswith("2-aminoethyl 3-O-[")
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    result = opsin_roundtrip_check(GIANT_AMINOETHYL_TRIDECASACCHARIDE, name)
    assert result["passed"] is True, (name, result)
    assert _full_inchikey(result["opsin_smiles"]) == _full_inchikey(
        GIANT_AMINOETHYL_TRIDECASACCHARIDE
    )


def test_branched_with_capped_aminopentyl_root_uronic_branches():
    # v33 Task 1.3 SCALE witness: an 8-unit tree (two glucuronic-acid
    # termini) behind an aminopentyl-capped root. Also the witness for the
    # `_glycoside_cap_name` attach-locant fix: the OLD converter guessed
    # `attach_locant=1` against "pentan-1-amine" (wrong end) and silently
    # declined; `_located_acyclic_alkyl_name` derives `5-aminopentyl` instead.
    mol = Chem.MolFromSmiles(AMINOPENTYL_URONIC_BRANCHED_OCTASACCHARIDE)
    assert mol is not None
    name = O.name_disaccharide(mol)
    assert name is not None, "expected a name, got an abstention"
    assert name.startswith("5-aminopentyl 4-O-[")
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    result = opsin_roundtrip_check(AMINOPENTYL_URONIC_BRANCHED_OCTASACCHARIDE, name)
    assert result["passed"] is True, (name, result)
    assert _full_inchikey(result["opsin_smiles"]) == _full_inchikey(
        AMINOPENTYL_URONIC_BRANCHED_OCTASACCHARIDE
    )


def test_branched_with_capped_ester_linker_root():
    # v33 Task 1.3 SCALE witness: a simple branched hexasaccharide capped
    # with a methyl-ester-terminated hexyl linker (a non-amine functionalized
    # cap, exercising `_located_acyclic_alkyl_name`'s general FG path rather
    # than the amine-specific branch).
    mol = Chem.MolFromSmiles(METHYL_ESTER_HEXYL_HEXASACCHARIDE)
    assert mol is not None
    name = O.name_disaccharide(mol)
    assert name is not None, "expected a name, got an abstention"
    assert name.startswith("6-methoxy-6-oxohexyl ")
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    result = opsin_roundtrip_check(METHYL_ESTER_HEXYL_HEXASACCHARIDE, name)
    assert result["passed"] is True, (name, result)
    assert _full_inchikey(result["opsin_smiles"]) == _full_inchikey(
        METHYL_ESTER_HEXYL_HEXASACCHARIDE
    )


def test_branched_capped_root_declines_when_cap_is_a_ring():
    # Scope boundary (unchanged): a CYCLIC aglycone cap is out of scope for
    # this simple alkyl/aryl relaxation (mirrors _oligo_topology exactly) --
    # must still fail closed, never fabricate a ring-substituent name.
    smi = (
        "OC[C@H]1O[C@H](OC[C@H]2OC(OC3CCCCC3)[C@H](O)[C@@H](O)[C@@H]2O"
        "[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"
    )
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    assert O.name_branched_oligosaccharide(mol) is None


def test_maltotriose_and_branched_glucotriose_unaffected():
    # PIN-safety regression guard: an EXISTING free-reducing-end linear chain
    # (maltotriose, name_linear_oligosaccharide's job) and the existing
    # free-reducing-end BRANCHED_GLUCOTRIOSE positive above must stay
    # byte-identical -- this task only ADDS a capped-root relaxation, never
    # touches the free-hemiacetal path.
    maltotriose = (
        "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O[C@H]3[C@H](O)"
        "[C@@H](O)C(O)O[C@@H]3CO)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O"
    )
    m1 = Chem.MolFromSmiles(maltotriose)
    assert O.name_disaccharide(m1) is not None
    m2 = Chem.MolFromSmiles(BRANCHED_GLUCOTRIOSE)
    assert O.name_branched_oligosaccharide(m2) == BRANCHED_GLUCOTRIOSE_NAME
