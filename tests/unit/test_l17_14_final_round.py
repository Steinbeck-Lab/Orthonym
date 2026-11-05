"""lane/l17 final round: process-wide state that moved a name with the history of the process, and the verdict of the
constitution oracle under load. Each test pins the behaviour the round restored."""
import pytest
from rdkit import Chem

from orthonym.assembly import memo

pytestmark = pytest.mark.unit


def test_the_split_probe_name_is_not_kept_across_namings(monkeypatch):
    """(A) ``group_splitting._probe_full_split_name`` re-names the molecule with the split on. A process-wide cache keyed by
    the SMILES alone served the name of one naming (its tier flags, its state) to the next naming of the molecule, so the
    second call of a molecule named fewer substituent fragments than the first
    (``tests/integration/test_producer_identity.py::test_memo_decisions_are_equal[row96]``). The nested ``name_compound``
    call is memoised per naming scope, keyed by the ambient state: each naming asks it afresh."""
    from orthonym import namer
    from orthonym.assembly import group_splitting
    asked = []
    monkeypatch.setattr(namer, "name_compound", lambda smi, **kw: asked.append(smi) or "ethyl acetate")
    mol = Chem.MolFromSmiles("CCOC(C)=O")
    for _ in range(2):
        token = memo.push_scope()                  # the scope of one top-level naming call
        try:
            assert group_splitting._probe_full_split_name(mol) == "ethyl acetate"
        finally:
            memo.pop_scope(token)
    assert len(asked) == 2


def _unavailable(unread):
    from orthonym.validation import atom_coverage
    return lambda m, name, *a, **k: atom_coverage.CoverageResult(
        total_heavy_atoms=m.GetNumHeavyAtoms(), claimed_atoms=0, unclaimed_atoms=m.GetNumHeavyAtoms(),
        coverage_ratio=0.0, is_complete=False, method="unavailable", opsin_unread=unread)


def test_an_oracle_run_that_gave_no_answer_gives_no_verdict(monkeypatch):
    """(F1) ``name_constitution_verdict``: ``method == "unavailable"`` of a run that gave no answer (a timeout of the JVM on
    a loaded host, a Java-side error) is ``None``, not ``False``. ``names_another_molecule`` rejects a pipeline name on
    ``False`` only, so the name a molecule gets does not depend on the load of the host; a name OPSIN answered with no
    structure stays ``False`` (``opsin_unread``: a property of the name)."""
    from orthonym.assembly.ester_prefix_probe import names_another_molecule
    from orthonym.decomposition import engine
    from orthonym.metrics.provenance import wider_tier_ctx
    from orthonym.validation import atom_coverage
    mol = Chem.MolFromSmiles("CCOC(=O)c1ccccc1")
    engine._PROVEN_COMPLETE_CACHE.clear()
    token = wider_tier_ctx.set(True)
    try:
        monkeypatch.setattr(atom_coverage, "validate_atom_coverage", _unavailable(False))
        assert engine.name_constitution_verdict("ethyl benzoate", mol) is None
        assert not names_another_molecule("ethyl benzoate", mol)
        monkeypatch.setattr(atom_coverage, "validate_atom_coverage", _unavailable(True))
        assert engine.name_constitution_verdict("(acetyloxy) methyl benzoate", mol) is False
        assert names_another_molecule("(acetyloxy) methyl benzoate", mol)
    finally:
        wider_tier_ctx.reset(token)
        engine._PROVEN_COMPLETE_CACHE.clear()


def test_the_coverage_oracle_tells_a_run_without_an_answer_from_an_unread_name(monkeypatch):
    """(F1) ``validate_atom_coverage``: a name OPSIN read as no structure is ``opsin_unread``; a run that gave no answer
    (``atom_coverage._opsin_read`` returns ``answered`` False: the subprocess timed out, a Java-side error) is not. The
    parse cache keeps the answer of the first run of a name, so a later call says the same."""
    from orthonym.validation import atom_coverage as ac
    mol = Chem.MolFromSmiles("CCO")
    monkeypatch.setattr(ac, "find_opsin_jar", lambda: "/fake.jar")
    monkeypatch.setattr("shutil.which", lambda *_: "/usr/bin/java")
    ac._OPSIN_PARSE_CACHE.clear()
    ac._OPSIN_UNANSWERED.clear()
    reads = {"unreadable name": (None, True), "timed out name": (None, False), "ethanol": ("CCO", True)}
    monkeypatch.setattr(ac, "_opsin_read", lambda name, jar, allow_radicals=False: reads[name])
    try:
        unread = ac.validate_atom_coverage(mol, "unreadable name")
        assert unread.method == "unavailable" and unread.opsin_unread
        failed = ac.validate_atom_coverage(mol, "timed out name")
        assert failed.method == "unavailable" and not failed.opsin_unread
        assert not ac.validate_atom_coverage(mol, "timed out name").opsin_unread          # served from the parse cache
        ok = ac.validate_atom_coverage(mol, "ethanol")
        assert ok.method == "parse_back" and ok.is_complete and not ok.opsin_unread
    finally:
        ac._OPSIN_PARSE_CACHE.clear()
        ac._OPSIN_UNANSWERED.clear()


# (B) the acyclic ester on the ring alcohol of a dilactone (antimycin-type): the lactones are class 16, so the ester is the
# principal group
_ANTIMYCIN = {
    "CCCC(=O)O[C@H]1[C@H](C)OC(=O)[C@@H](NC(=O)c2cccc(NC=O)c2O)[C@@H](C)OC(=O)[C@@H]1CCC(C)C":
        "(2R,3S,6S,7R,8R)-3-(3-formamido-2-hydroxybenzamido)-2,6-dimethyl-8-(3-methylbutyl)-4,9-dioxo-1,5-dioxonan-7-yl butanoate",
    "CCCCCC[C@H]1C(=O)O[C@H](C)[C@H](NC(=O)c2cccc(NC(C)=O)c2O)C(=O)O[C@@H](C)[C@@H]1OC(=O)[C@@H](C)CC":
        "(2R,3S,6S,7R,8R)-3-(3-acetamido-2-hydroxybenzamido)-8-hexyl-2,6-dimethyl-4,9-dioxo-1,5-dioxonan-7-yl (2S)-2-methylbutanoate",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles", sorted(_ANTIMYCIN))
def test_an_acyclic_ester_on_a_dilactone_ring_is_the_pin(smiles, tier):
    """(B) SENIORITY ORDER FOR CLASSES, Table 4.1 (the Blue Book), "9 Esters (functional class names are given to
    noncyclic esters; lactones and other cyclic esters are named as heterocycles; see 16 below)": the acyclic ester is the
    principal group, the lactones are ring 'dioxo' and the amide a prefix (class 11 below class 9). The probe of the
    decomposition engine at the PIN tier met the cascade's prefix name of the ester and set the one-way label of the call
    (``record_general_ring_prefix``), so the split's name, right by the book, shipped as ``pin_unverified``."""
    from orthonym.cli import _tiered_namer
    from tests.support.rt_assert import assert_full_rt
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert row["name"] == _ANTIMYCIN[smiles], row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


# (C1) a pseudoester ranks as an ester, by its acid: the senior acid is the parent of the name
_PSEUDOESTERS = {
    "C[Si](C)(C)OC(=O)c1ccc(OC(C)=O)cc1": "trimethylsilyl 4-(acetyloxy)benzoate",
    "O=C(Oc1ccc(C(=O)O[Si](C)(C)C)cc1)c1ccccc1": "trimethylsilyl 4-(benzoyloxy)benzoate",
    "CCCC[Sn](CCCC)(CCCC)OC(=O)c1ccc(OC(C)=O)cc1": "tributylstannyl 4-(acetyloxy)benzoate",
    "CSC(=O)CCCCC(=O)O[Si](C)(C)C": "trimethylsilyl 6-(methylsulfanyl)-6-oxohexanoate",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles", sorted(_PSEUDOESTERS))
def test_a_pseudoester_of_the_senior_acid_is_the_parent(smiles, tier):
    """(C1) "Definitions" (the Blue Book): "Pseudoesters are ranked as esters in the seniority order of
    classes (see "; class 9 of Table 4.1 (:18182) ranks the esters by their acid and "Esters cited as
    prefixes" (:31696) cites the ester of the junior acid as a prefix of the senior one's name ('ethyl 2-[(ethoxycarbonyl)
    oxy]-4,4-dimethyl-3-oxopentanoate (PIN)', the example of an ester cited beside a senior group): benzoic acid (a ring) is
    senior to acetic acid and the trimethylsilyl ester of the O acid is senior to the S-ester of its thioic analogue (class
    7a,:18172, "Chalcogen analogues follow each of the corresponding oxygen acids"). The pseudoester ranked below every
    ester of ``SENIORITY_ORDER``, so the name was '4-{oxo[(trimethylsilyl)oxy]methyl}phenyl acetate' (the methyl analogue,
    'methyl 4-(acetyloxy)benzoate', was right) at both tiers as a PIN."""
    from orthonym.cli import _tiered_namer
    from tests.support.rt_assert import assert_full_rt
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert row["name"] == _PSEUDOESTERS[smiles], row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_an_ester_group_spelled_oxo_alkoxy_methyl_is_no_pin_spelling():
    """(C1) "Esters cited as prefixes" (the Blue Book): '(alkyloxy)carbonyl', '(alkanyloxy)carbonyl' for
    -CO-OR' on a carbon, 'alkoxy...oxo' on a chain; 'oxo[(trimethylsilyl)oxy]methyl' and '(methylsulfanyl)(oxo)methyl' are
    in no PIN (the book has neither spelling)."""
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    for name in ("4-{oxo[(trimethylsilyl)oxy]methyl}phenyl acetate", "methyl 4-{oxo[(tributylstannyl)oxy]methyl}benzoate",
                 "4-[oxo(propan-2-yloxy)methyl]phenyl acetate", "4-[(methylsulfanyl)(oxo)methyl]phenyl acetate"):
        assert non_pin_vocabulary(name), name
    for name in ("trimethylsilyl 4-(acetyloxy)benzoate", "methyl 4-(methoxycarbonyl)benzoate",
                 "methyl 4-{2-oxo-2-[(trimethylsilyl)oxy]ethyl}benzoate", "2-oxo-2-(propan-2-yloxy)ethyl acetate"):
        assert not non_pin_vocabulary(name), name


# (C2) a partial ester of a phosphorus or sulfur oxoacid is an ester of class 9 ranked by its acid (7d), below the carbamate
# (7b) and the carboxylic ester (7a)
_PARTIAL_ESTERS = {
    "COC(=O)Nc1ccc(OP(=O)(O)O)cc1": "methyl [4-(phosphonooxy)phenyl]carbamate",
    "CCOC(=O)Nc1ccc(OS(=O)(=O)O)cc1": "ethyl [4-(sulfooxy)phenyl]carbamate",
    "COC(=O)CCOS(=O)(=O)O": "methyl 3-(sulfooxy)propanoate",
    "CC(=O)Nc1ccc(OP(=O)(O)O)cc1": "4-acetamidophenyl dihydrogen phosphate",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles", sorted(_PARTIAL_ESTERS))
def test_a_partial_ester_of_a_noncarbon_oxoacid_ranks_by_its_acid(smiles, tier):
    """(C2) Decided by the book, not by the free hydrogen: (the Blue Book) "The seniority order of anions
    is corresponding to that of acids (see seniority order of acids in ", (:35916) "Partial acid esters of
    polybasic acids are named by citing alkyl groups... followed by the word 'hydrogen'" (an ester name) and
    (:55180), the book's phosphatidylethanolamine named on its carboxylic esters with the phosphate in a prefix. The
    carbamate (class 7b acid) and the carboxylic ester (7a) are the parent beside the partial ester of a class 7d acid;
    the amide (class 11) is below it. The three producers (``seniority._promoted_oxoacid_ester``,
    ``handlers.carbamate._carbamate_ranks_above_principal``, ``seniority.ESTER_ACID_ORDER``) read one order."""
    from orthonym.cli import _tiered_namer
    from tests.support.rt_assert import assert_full_rt
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert row["name"] == _PARTIAL_ESTERS[smiles], row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_the_orders_of_the_esters_of_the_noncarbon_acids_agree():
    """(C2) The partial esters are the esters of the class 7d acids in ``ESTER_ACID_ORDER``, the groups the carbamate handler
    ranks itself above are exactly those of a lower rank than its own, and a partial ester of the order is one the offer of
    ``_promoted_oxoacid_ester`` can rank."""
    from orthonym.rules.seniority import (_OXOACID_ESTERS_PROMOTED, ESTER_ACID_ORDER, ester_acid_rank)
    carbamate = ester_acid_rank("carbamate")
    for partial in ("phosphate_monoester", "phosphate_diester", "sulfate_monoester"):
        assert ester_acid_rank(partial) > carbamate and ester_acid_rank(partial) > ester_acid_rank("ester"), partial
        assert partial in _OXOACID_ESTERS_PROMOTED, partial
    assert [label for label, _ in ESTER_ACID_ORDER] == ["7a", "7b", "7c", "7d"]


# (I1) the principal chain follows the senior amide: the parent of two amides on different chains is named, not dropped
_TWO_AMIDES = {
    "CC(C)C(=O)NCC(=O)NC(C)C": "2-methyl-N-{2-oxo-2-[(propan-2-yl)amino]ethyl}propanamide",
    "C/C=C/C(=O)NCC(=O)NCC": "(2E)-N-[2-(ethylamino)-2-oxoethyl]but-2-enamide",
    "C=CC(=O)NCCC(=O)NC": "N-[3-(methylamino)-3-oxopropyl]prop-2-enamide",
    "CCC(C)C(=O)NCCC(=O)NC": "2-methyl-N-[3-(methylamino)-3-oxopropyl]butanamide",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles", sorted(_TWO_AMIDES))
def test_the_senior_amide_of_two_is_the_parent_with_its_chain(smiles, tier):
    """(I1) (a) (the Blue Book) and (:20938), ``rules.amide_parent``: of the two amide groups the senior
    parent is the acyl with the longer chain (isobutyramide before acetamide), at the same length the one with the multiple
    bond (prop-2-enamide before propanamide). ``classify_parts.chain_parent`` counted the amide groups of the class on each
    chain, so the glycyl chain that bears its own amide and the nitrogen of the other won the principal chain, and the amide
    assembler named the acyl from it ('N-[2-(methylamino)-2-oxoethyl]propanamide': the 2-methyl lost, the crotonyl chain
    named 'acetamide'); the read-back rejected those names, the junior-parent split was declined by the lane, and the
    molecule shipped with no suffix ('1-(2-methylpropanamido)-2-oxo-2-[(propan-2-yl)amino]ethane')."""
    from orthonym.cli import _tiered_namer
    from tests.support.rt_assert import assert_full_rt
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert row["name"] == _TWO_AMIDES[smiles], row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_the_principal_chain_is_cut_to_the_senior_amide_only_when_the_choice_left_some_out():
    from orthonym.classify_parts import _functional_groups_of_the_parent
    from types import SimpleNamespace
    both = [(6, 7, 5, 4), (2, 3, 1, 0)]
    groups = {"secondary_amide": both, "ketone": [(1, 2, 3)]}
    cut = _functional_groups_of_the_parent(SimpleNamespace(
        functional_groups=groups, principal_group="secondary_amide", principal_group_atoms=[both[0]]))
    assert cut["secondary_amide"] == [both[0]] and cut["ketone"] == groups["ketone"] and groups["secondary_amide"] == both
    same = _functional_groups_of_the_parent(SimpleNamespace(
        functional_groups=groups, principal_group="secondary_amide", principal_group_atoms=both))
    assert same is groups
    other = _functional_groups_of_the_parent(SimpleNamespace(
        functional_groups={"primary_alcohol": [(1, 2)] * 2}, principal_group="primary_alcohol",
        principal_group_atoms=[(1, 2)]))
    assert other == {"primary_alcohol": [(1, 2)] * 2}


# (I2) the ester of carbamic acid and of its chalcogen analogues is a class 9 ester: no PIN cites it in a prefix beside an amide
_CARBAMOTHIOATE_UNDER_AMIDE = ["CNC(=S)OC1CCC(N(C)C(C)=O)CC1", "CNC(=S)OCCNC(C)=O", "CN(C)C(=S)SCCNC(C)=O"]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", _CARBAMOTHIOATE_UNDER_AMIDE)
def test_no_pin_cites_a_carbamothioate_beside_an_amide(smiles):
    """(I2) SENIORITY ORDER FOR CLASSES, Table 4.1 (the Blue Book,:18184): esters (class 9) are senior to amides
    (class 11), and "Carbonic acid" (:30756) makes the ester of carbamothioic acid one of them: the PIN tier named
    'N-{2-[(methylcarbamothioyl)oxy]ethyl}acetamide' (the group cited in a prefix) as a PIN while the wider tiers gave the
    ester. The PIN tier has no name of the ester for these molecules and gives none."""
    from orthonym.cli import _tiered_namer
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert not row["is_pin"] and row["tier"] != "pin_verified", row
    best = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert best["name"] and "amide" not in best["name"].split()[-1], best


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_the_carbamate_is_senior_to_the_carbamothioate_and_spelled_as_a_carbamate(tier):
    """(I2) 'CNC(=O)Oc1ccc(SC(=O)N(C)C)cc1': the ester of carbamic acid and the S-ester of carbamothioic acid; the oxygen acid
    is senior class 7a, the Blue Book: "Chalcogen analogues follow each of the corresponding oxygen
    acids"), so the carbamate is the parent and the S-ester a prefix. The thioester perception reads on the carbamothioate had
    made the molecule a chalcogen ester whose acid word was '(dimethylamino)methanethioate',:30756: "Substitution
    of formic acid is not recommended"), at the PIN tier."""
    from orthonym.cli import _tiered_namer
    from tests.support.rt_assert import assert_full_rt
    smiles = "CNC(=O)Oc1ccc(SC(=O)N(C)C)cc1"
    row = _tiered_namer("pin", tier).name_tiered(smiles)
    assert row["name"] == "4-[(dimethylcarbamoyl)sulfanyl]phenyl methylcarbamate", row
    assert row["tier"] == "pin_verified" and row["is_pin"], row
    assert_full_rt(row["name"], smiles)


def test_the_vocabulary_reads_the_carbamic_esters_and_the_substituted_formic_acid():
    from orthonym.rules.pin_vocabulary import non_pin_vocabulary
    for name in ("N-{2-[(methylcarbamothioyl)oxy]ethyl}acetamide", "2-[(methylcarbamoyl)oxy]ethan-1-ol",
                 "N-{2-[(dimethylcarbamothioyl)sulfanyl]ethyl}acetamide",
                 "S-{4-[(dimethylcarbamoyl)sulfanyl]phenyl} (dimethylamino)methanethioate"):
        assert non_pin_vocabulary(name), name
    for name in ("S-phenyl dimethylcarbamothioate", "4-[(dimethylcarbamoyl)sulfanyl]phenyl methylcarbamate",
                 "N-carbamoylacetamide", "4-carbamoylphenol"):
        assert not non_pin_vocabulary(name), name


# (F2) the pseudoesters of boron: the group 'boranetriyl' and the one-cut fallback of the mixed hub esters
_BORON_AND_MIXED = {
    "CC(=O)OB(OC(C)=O)OC(C)=O": "boranetriyl triacetate",
    "CC(=O)OB(OC(C)=O)OC(=O)c1ccccc1": "(acetyloxy)(benzoyloxy)boranyl acetate",
    "CC(=O)O[Si](C)(OC(C)=O)OC(=O)c1ccccc1": "(acetyloxy)(benzoyloxy)(methyl)silyl acetate",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", sorted(_BORON_AND_MIXED))
def test_the_pseudoesters_of_a_hub_with_three_acyloxy_groups_keep_a_name_below_the_pin(smiles):
    """(F2) "Definitions" (the Blue Book) lists B among the elements Z of a pseudoester, (:2056) among
    the elements of the parent hydrides of PINs, and 'boranetriyl' is a group of the book,:37266);
    (:31819) names the identical anions of one multivalent group by multiplicative nomenclature ('boranetriyl triacetate'),
     (:31831) the different anions by method (1) (not built) or method (2), "acceptable in general
    nomenclature" ('(acetyloxy)(benzoyloxy)boranyl acetate'). The book prints no pseudoester of boron, so the name is below the
    PIN. The lane declined the one-cut name of a hub whose groups include identical ones, so the best-effort name of
    B(OAc)3 fell to the general engine's '1-{[bis(acetyloxy)boranyl]oxy}-1-oxoethane'."""
    from orthonym.cli import _tiered_namer
    from tests.support.rt_assert import assert_full_rt
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["name"] == _BORON_AND_MIXED[smiles], row
    assert row["tier"] == "systematic_verified" and not row["is_pin"], row
    assert_full_rt(row["name"], smiles)
    assert not _tiered_namer("pin", "pin").name_tiered(smiles)["is_pin"]


def test_the_peptide_reading_of_the_molecule_asked_is_made_once_per_naming_call(monkeypatch):
    """(F4) ``n_acyl_amido.in_peptide`` is asked for every chain substituent a call names: it re-parsed the SMILES of the molecule
    the caller asked about and matched the dipeptide pattern on it each time. It reads it once per naming scope."""
    from orthonym.assembly import fragment_naming, memo, n_acyl_amido
    from orthonym.rules import peptides
    calls = []
    real = peptides.peptide_like
    monkeypatch.setattr(peptides, "peptide_like", lambda mol: calls.append(mol.GetNumAtoms()) or real(mol))
    monkeypatch.setattr(fragment_naming, "outermost_name_input", lambda: "CC(=O)NCC(=O)NCC(=O)O")
    fragment = Chem.MolFromSmiles("CCO")
    token = memo.push_scope()
    try:
        answers = [n_acyl_amido.in_peptide(fragment) for _ in range(4)]
    finally:
        memo.pop_scope(token)
    assert answers == [True] * 4
    assert calls.count(Chem.MolFromSmiles("CC(=O)NCC(=O)NCC(=O)O").GetNumAtoms()) == 1, calls
    token = memo.push_scope()
    try:
        n_acyl_amido.in_peptide(fragment)               # the next call asks again
    finally:
        memo.pop_scope(token)
    assert calls.count(Chem.MolFromSmiles("CC(=O)NCC(=O)NCC(=O)O").GetNumAtoms()) == 2, calls


# (F5) (l) read literally against the parent that chose
_L_DIFFERS = ["CCC(=O)N(C)[C@@H](C)C(=O)N", "CCC(=O)N(C)[C@H](C)C(=O)NC"]
_L_AGREES = ["CCC(=O)N[C@@H](C)C(=O)NCC", "CCC(=O)NCC(=O)N[C@@H](C)CC", "CCC(=O)N(C)CC(=O)N", "C[C@H](CC)C(=O)NCC(=O)N",
             "CCC(=O)N(C)C(C)C(=O)N"]       # the last: the centre of the alanyl chain is not given


@pytest.mark.parametrize("smiles", _L_DIFFERS)
def test_a_parent_chosen_over_a_stereogenic_chain_of_the_same_rank_is_below_the_pin(smiles):
    """(F5) "The senior ring, ring system, or principal chain has one or more stereogenic centers [criterion (l)
    in ]" (the Blue Book): two chains that tie up to (k) (here two propanamide chains, no multiple bond, no
    isotope) and one of them stereogenic. The ruling of the lane reads (l) as the examples show it, so (the more
    prefixes,:21604) picks the N-methylpropanamide chain; the literal reading picks the alanyl chain, so the name is not
    shown to be the preferred one and is labelled below the PIN, at every tier."""
    from orthonym.cli import _tiered_namer
    for tier in ("pin", "best-effort"):
        row = _tiered_namer("pin", tier).name_tiered(smiles)
        assert not row["is_pin"] and row["tier"] != "pin_verified", row
    best = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert best["name"].endswith("propanamide") and "N-methyl" in best["name"], best
    assert "P-44.4.1.12" in str(best.get("spelling_failures")), best


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", _L_AGREES)
def test_a_parent_that_both_readings_choose_keeps_its_label(smiles):
    """(F5) The parent is the stereogenic chain, or no chain ties with it: (l) read literally does not differ, so the label is
    the one the criteria gave."""
    from orthonym.cli import _tiered_namer
    row = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert row["tier"] == "pin_verified" and row["is_pin"], row


def test_the_literal_reading_of_l_is_decided_on_the_parent_and_the_tied_chains_only():
    from orthonym.rules.amide_parent import comparable_amide_units, stereo_criterion_could_differ
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCC(=O)N(C)[C@@H](C)C(=O)N")
    carbonyls = {m[0]: cls for cls, m in comparable_amide_units(mol)}
    tertiary = next(c for c, cls in carbonyls.items() if cls == "tertiary_amide")
    primary = next(c for c, cls in carbonyls.items() if cls == "primary_amide")
    assert stereo_criterion_could_differ(mol, {tertiary}) and not stereo_criterion_could_differ(mol, {primary})
    for smiles in _L_AGREES:
        m = Chem.MolFromSmiles(smiles)
        flat = comparable_amide_units(m)
        if flat is None:
            continue
        # the senior parent of the choice (the key of ``senior_carboxamide_group``) never differs from (l) in these rows
        from orthonym.rules.amide_parent import amide_parent_key
        best = min((amide_parent_key(m, mm[0]), mm[0]) for _, mm in flat)[1]
        assert not stereo_criterion_could_differ(m, {best}), smiles
