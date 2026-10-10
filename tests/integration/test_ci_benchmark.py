"""
CI Benchmark Test Suite -- 100 compounds from ChEBI 500-sample (seed=123).

Deterministic regression test: verifies that name_compound produces
the exact same IUPAC name for 100 representative ChEBI compounds.
Any change in output indicates a naming regression.

Sampling:
  1. Load ChEBI_SMILES.txt, deduplicate by canonical SMILES (111,823 unique)
  2. random.Random(123).sample(smiles_list, 500) -> 500-sample
  3. sample[:100] -> CI benchmark subset

A row that holds a plain string only verifies deterministic name output against the
pre-computed expected name (the OPSIN validity gate is off, as in most of the suite).
A row that holds a ``_TierRow`` is asserted as the tier contract in production state
(the gate on, marker ``opsin_gate``), because the snapshot predates the default tier's
emission rule: see the comment above ``_TierRow``.

Target: all 100 tests pass in <60 seconds.
"""

import re

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym import name_compound
from orthonym.errors import is_failure_name
from tests.support.default_tier import (
    declined_at_default,
    default_tier_row,
    strict_path_name,
)
from tests.support.rt_assert import (
    _full_inchikey,
    _independent_parse,
    assert_full_rt,
    assert_tier_contract,
    name_best_effort,
)

# Tier-contract rows: the rows whose February 2026 byte snapshot is replaced.
#
# The snapshot was recorded with the OPSIN validity gate off and before the default
# tier's emission rule. Since 4e4e7cc52 (user decision 2026-09-30; the submitted paper,
# Methods, "Tiers": "The default configuration emits a name only when the pipeline can
# build the preferred IUPAC name (PIN); otherwise, it declines.") the default tier ships a
# name only when the strict PIN path built and verified it, and declines every other
# name with a reason code; the wider tiers keep every name (tests/support/default_tier.py).
# Most old values were also not the structure's name (an OPSIN read-back gives another
# molecule or none), so none is restored. A row of this kind runs in production state
# (the gate on) and every name it asserts is read back by a FRESH OPSIN call to the
# input's full InChIKey (tests/support/rt_assert.py), never by the engine's own gate:
#
# _ships(name) the default tier ships exactly ``name``; best-effort
# names the molecule, round-trip exact (never worse). A
# Chapter name (furostan, an oligosaccharide, the AMP
# hydrate) is shipped by the earlier decision recorded in
# test_regression_v5.py: (the Blue Book)
# identifies no PIN there, so none is claimed here.
# _declined_at_default(name) the default tier declines (NO_VERIFIED_PIN); the strict
# path builds exactly ``name`` and the best-effort tier
# gives it; both read back exactly. ``name`` is pinned as
# the strict path's output, not claimed to be the PIN.
# _declined(code) the default tier declines (``code`` when given) and the
# best-effort tier names the molecule, round-trip exact.
# _out_of_scope(sentinel, c) no tier names it: the default tier returns ``sentinel``
# with limit code ``c``; a wider tier must never be wrong.
# _tier_contract best-effort names the molecule, round-trip exact; the
# default tier declines or ships a name that round-trips
# too (assert_tier_contract).
# _constitution_only(name) the default tier ships ``name`` on the constitution-only
# comparison (OPSIN cannot assign its CIP labels).
_GATE = pytest.mark.opsin_gate


class _TierRow:
    """The expected outcome of one tier-contract row (see the comment above)."""

    def __init__(self, kind, name=None, code=None, strict_name_must_round_trip=False,
                 best_effort_same=True):
        self.kind = kind
        self.name = name
        self.code = code
        self.strict_name_must_round_trip = strict_name_must_round_trip
        self.best_effort_same = best_effort_same

    def __repr__(self):
        return f"{self.kind}({self.name or self.code or ''})"

    def check(self, smiles):
        getattr(self, "_check_" + self.kind)(smiles)

    def _regression(self, smiles, got):
        return (
            f"CI REGRESSION: {smiles}\n"
            f"  Expected: {self.name}\n"
            f"  Got:      {got}"
        )

    def _check_ships(self, smiles):
        pin, _ = assert_tier_contract(smiles)
        assert pin == self.name, self._regression(smiles, pin)

    def _check_tier_contract(self, smiles):
        assert_tier_contract(smiles)

    def _check_declined_at_default(self, smiles):
        declined_at_default(smiles, self.name, best_effort_same=self.best_effort_same)

    def _check_declined(self, smiles):
        row = default_tier_row(smiles)
        assert row["tier"] == "abstain" and is_failure_name(row["name"]), (
            f"the default tier must decline {smiles}, got {row}"
        )
        if self.code:
            assert row["limit_code"] == self.code, row
        pin, _ = assert_tier_contract(smiles)
        assert is_failure_name(pin), self._regression(smiles, pin)
        if self.strict_name_must_round_trip:
            assert_full_rt(strict_path_name(smiles), smiles, "strict path: ")

    def _check_out_of_scope(self, smiles):
        row = default_tier_row(smiles)
        assert (row["name"], row["tier"], row["limit_code"]) == (
            self.name, "abstain", self.code,
        ), row
        assert name_compound(smiles) == self.name, self._regression(
            smiles, name_compound(smiles))
        # no tier names it; a wider tier that ever does must not be wrong
        be = name_best_effort(smiles)
        if be.get("name") and be.get("source") != "abstain":
            assert_full_rt(be["name"], smiles, "best-effort: ")

    # The stereocentres of the one row that uses _check_constitution_only: the SMARTS of
    # each (its first atom) and the CIP label its descriptor in the asserted name gives it.
    _CIP_OF_NAME = (
        ("[C;R]([OX2H])(C(=O)[OX2H])", "S"),   # (1S)
        ("[CH1;R](OC(=O)C=C)", "S"),            # (4S)
        ("[CH1;R]([OX2H])", "R"),               # (3R) and (5R)
    )

    def _check_constitution_only(self, smiles):
        name = name_compound(smiles)
        assert name == self.name, self._regression(smiles, name)
        # OPSIN 2.9.0 reports "Failed to assign CIP stereochemistry... limitation in
        # OPSIN" for this name, so the full-key read-back is not available. Check the two
        # halves independently: (1) the name without its stereodescriptors reads back, by
        # a fresh OPSIN call, to the input's constitution (first InChIKey block);
        flat = re.sub(r"\((?:\d+[RSEZ],?)+\)-", "", name)
        opsin_smi = _independent_parse(flat)
        assert opsin_smi and (
            _full_inchikey(opsin_smi).split("-")[0]
            == _full_inchikey(smiles).split("-")[0]
        ), f"{flat!r} does not read back to the constitution of {smiles}"
        # (2) the CIP labels RDKit's rdCIPLabeler (not the engine's) gives the input's
        # stereocentres and its double bond are the name's descriptors.
        mol = Chem.MolFromSmiles(smiles)
        rdCIPLabeler.AssignCIPLabels(mol)
        for smarts, label in self._CIP_OF_NAME:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
            assert matches, smarts
            for match in matches:
                got = mol.GetAtomWithIdx(match[0]).GetProp("_CIPCode")
                assert got == label, (
                    f"{name}: RDKit gives {got} where the name says {label} "
                    f"({smarts})"
                )
        ene = mol.GetSubstructMatch(Chem.MolFromSmarts("c[CH1]=[CH1]C(=O)O"))
        bond = mol.GetBondBetweenAtoms(ene[1], ene[2])
        assert bond.GetProp("_CIPCode") == "E", f"{name}: the C=C is not (2E)"


def _ships(name):
    return _TierRow("ships", name)


def _tier_contract():
    return _TierRow("tier_contract")


def _declined_at_default(name, best_effort_same=True):
    return _TierRow("declined_at_default", name, best_effort_same=best_effort_same)


def _declined(code=None, strict_name_must_round_trip=False):
    return _TierRow(
        "declined", code=code, strict_name_must_round_trip=strict_name_must_round_trip)


def _out_of_scope(sentinel, code):
    return _TierRow("out_of_scope", sentinel, code)


def _constitution_only(name):
    return _TierRow("constitution_only", name)


# Pre-computed expected names for 100 ChEBI compounds (seed=123, first 100 of 500).
# Generated on 2026-02-07 after a phase Plans 01-04.
# Any change here indicates a naming regression that must be investigated.
CI_BENCHMARK = [
    pytest.param(
        "*=CC",
        # A wildcard atom is declined at every tier (limit code WILDCARD_ATOMS, 116296a0d); the
        # old expected 'ethane' dropped the wildcard atom, a different structure.
        _out_of_scope("compound with wildcard atoms (not supported)", "WILDCARD_ATOMS"),
        marks=_GATE,
    ),
    # L4 (TRIAGE ester-glue): the old '(2S,3R,5R,6R,8S)-17-phenylheptadecyl acetate' is a wrong
    # molecule (a C17 chain). The ester producer used to glue 'acetate' after a name whose
    # alcohol O was a 'hydroxy' prefix ('...decyl acetate'; OPSIN reads it as C25H30O13, the
    # input is C25H30O12).
    # It now builds the group word from the alcohol's structure; the PIN tier does not
    # certify that name (NO_VERIFIED_PIN), so the row asserts the tier contract.
    pytest.param(
        "CC(=O)OC[C@H]1O[C@@H](O[C@]23C[C@@H]4[C@@]2(COC(=O)c2ccccc2)"
        "[C@H]2O[C@]4(O)C[C@]3(C)O2)[C@H](O)[C@@H](O)[C@@H]1O",
        _tier_contract(),
        marks=_GATE,
    ),
    # a phase Plan 02 Task 03: estra ring system locant correction
    # 1,2,4-trien → 1,3,5-trien (canonical estra-1,3,5-triene numbering for
    # the aromatic A-ring per IUPAC. Per Plan 01 SUMMARY this is
    # "unrelated to a phase" (incidental locant correction, not a cascade
    # decision). Acceptable churn.
    # L5 (TRIAGE default-tier-pin-or-decline): the old text 'estra-1,3,5-trien-3,17-diol' is
    # bare estradiol, not the input (an estradiol carrying an N-butyl-N-methylundecanamide
    # chain). With the gate off the amide assembler took the steroid for the acyl chain and
    # cited '16,16-dihydroxy' on 'undecanamide' (stereo locants 12..16 on an 11-carbon
    # chain; OPSIN cannot parse it). It no longer counts ring carbons as chain atoms. The PIN
    # tier declines; best-effort names it exactly.
    pytest.param(
        "CCCCN(C)C(=O)CCCCCCCCCC[C@@H]1Cc2cc(O)ccc2[C@H]2CC[C@]3(C)"
        "[C@@H](O)CC[C@H]3[C@H]12",
        _tier_contract(),
        marks=_GATE,
    ),
    (
        "C[C@H](CCC(=O)O)[C@H]1C[C@H](O)[C@@]2(C)C3=CCC4C(C)(C)C(=O)"
        "CC[C@]4(C)C3=CC[C@]12C",
        # j7 (TRIAGE g2 G2-C6): the terminal -COOH is the '-24-oic acid' suffix, the
        # senior class, the Blue Book); the 3-one becomes '3-oxo'; the
        # C-9=C-11 bond takes the compound locant '9(11)' (1),:16634).
        # The old '...chola-7,9-dien-3,24-dione' read C-9=C-10 (OPSIN: no parse).
        # A name:50943: no PIN in Chapter, labelled best_effort;
        # OPSIN full-InChIKey exact.
        "(10S,13R,14R,15S,17R,20R)-15-hydroxy-4,4,14-trimethyl-3-oxochola-7,9(11)-dien-24-oic acid",
    ),
    pytest.param(
        "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC",
        _declined_at_default("(2S)-3-(decanoyloxy)-2-hydroxypropyl docosanoate"),
        marks=_GATE,
    ),
    # L4: a free acid owns the suffix, so the ester is the prefix of a partial ester of a
    # polybasic acid: "Partial esters of polybasic acids and their salts"
    # (the Blue Book) "Method (1) generates preferred IUPAC names." (:31940), method (1)
    # being "substitutively on the basis of the anion... the ester group(s) being cited as
    # prefixes"; '2-chloro-6-(ethoxycarbonyl)benzoic acid (PIN)' (:31950). The alkoxy part
    # of a branched alkyl is a compound prefix,:27665) cited in its own marks.
    # The old 'decyloxycarbonyl' read the carbon COUNT and named the isomer n-decyl.
    # OPSIN 2.9.0 full-InChIKey exact; pin_verified with the gate on.
    pytest.param(
        "CC(C)CCCCCCCOC(=O)c1ccccc1C(=O)O",
        _ships("2-{[(8-methylnonyl)oxy]carbonyl}benzoic acid"),
        marks=_GATE,
    ),
    pytest.param(
        "O=C(O)c1ccccc1-c1c2ccc(=O)c([As]3SCCS3)c-2oc2c([As]3SCCS3)"
        "c(O)ccc12",
        _declined("UNSUPPORTED_ELEMENT"),
        marks=_GATE,
    ),
    pytest.param(
        "Cc1cc([C@@]2(C)CCCC2(C)C)c(O)c(O)c1-c1c(C)cc([C@@]2(C)CCCC2"
        "(C)C)c(O)c1O",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3"
        "(C)[C@H]2[C@@H]1C",
        _ships("furostan"),
        marks=_GATE,
    ),
    pytest.param(
        "CC(C)[C@H](N)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](CCCCN)"
        "C(=O)O",
        _declined_at_default("valylarginyllysine"),
        marks=_GATE,
    ),
    pytest.param(
        "NC(=O)[C@H](CCC/N=C(/N)CF)NC(=O)c1ccccc1",
        _ships("(2S)-5-[(1-amino-2-fluoroethylidene)amino]-2-benzamidopentanamide"),
        marks=_GATE,
    ),
    pytest.param(
        "OCC1OC(Oc2cc(O)c3c(c2)OC(c2ccc(O)c(OC4OC(CO)C(O)C(O)C4O)c2)"
        "C(O)C3)C(O)C(O)C1O",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "CSCC[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCCCN)C(=O)O",
        _declined_at_default("lysylglutamylmethionine"),
        marks=_GATE,
    ),
    pytest.param(
        "CC(C)[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](Cc1ccccc1)"
        "C(=O)O",
        _declined_at_default("aspartylvalylphenylalanine"),
        marks=_GATE,
    ),
    # L5 (TRIAGE default-tier-pin-or-decline): the old text is a different molecule (OPSIN
    # round trip: not the input; it has no 4-hydroxyphenyl and a 'hexadecyl'). With the gate
    # off the decomposition engine glued 'ethane-1,2-diyl' in front of the macrocycle (the
    # ethylenediamine fragment turned into a diyl, a nitrogen lost; OPSIN cannot parse it);
    # since 4e4e7cc52 the default tier declines and best-effort names it exactly.
    pytest.param(
        "CC[C@H](C)C[C@H](C)CCCCCCCCC(=O)N[C@H]1C[C@@H](O)[C@@H]"
        "(NCCN)NC(=O)[C@@H]2[C@@H](O)CCN2C(=O)[C@H]([C@H](O)CCN)"
        "NC(=O)[C@H]([C@H](O)[C@@H](O)c2ccc(O)cc2)NC(=O)[C@@H]2"
        "C[C@@H](O)CN2C(=O)[C@H]([C@@H](C)O)NC1=O",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "O=[C]O[O-]",
        _declined(),
        marks=_GATE,
    ),
    # L5 (TRIAGE default-tier-pin-or-decline): the old text 'icosyl acetate' names a fragment
    # (a C20 chain). With the gate off the polycyclic cage producer spelled the 2-acetyloxy
    # group by its carbon count, '2-ethoxy' (the C=O lost: OPSIN C22H28O8 for the input's
    # C22H26O9). The cage namer now spells a branch by its carbon count only for an unbranched
    # saturated alkyl. The PIN tier does not build this cage; best-effort names it exactly.
    pytest.param(
        "C=C1[C@@H](O)O[C@H]2[C@H]1C[C@@H](OC(C)=O)[C@]13C(=O)O"
        "[C@H]4C[C@](C)(O)[C@H]([C@H]41)[C@@]31C=C(C)[C@]2(O)O1",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C"
        "\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC"
        "/C(C)=C\\COP(=O)(O)OP(=O)(O)O",
        _ships(
            "(2Z,6Z,10Z,14Z,18Z,22Z,26Z,30Z,34Z,38E,42E)-3,7,11,15,19,23,27,31,35,39,43,47-"
            "dodecamethyloctatetraconta-2,6,10,14,18,22,26,30,34,38,42,46-dodecaen-1-yl trihydrogen diphosphate"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "N[C@@H](CC(=O)O)C(=O)NCC(=O)N[C@@H](Cc1c[nH]c2ccccc12)"
        "C(=O)O",
        _declined_at_default("aspartylglycyltryptophan"),
        marks=_GATE,
    ),
    pytest.param(
        "O=c1c(O[C@@H]2OC(CO)[C@@H](O)[C@H](O)C2O[C@@H]2OC(CO)"
        "[C@H](O)[C@H](O)C2O[C@@H]2OC(CO)[C@@H](O)[C@H](O)C2O)"
        "c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "CC[C@H](C)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CC(N)=O)"
        "[C@@H](C)O)C(=O)O",
        _declined_at_default("asparaginylthreonylisoleucine"),
        marks=_GATE,
    ),
    pytest.param(
        "CCCCCC(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H]1C(=O)N[C@@H]"
        "(CCCN=C(N)N)C(=O)N[C@H]2CC[C@@H](O)N(C2=O)[C@@H]"
        "([C@@H](C)CC)C(=O)N(C)[C@@H](CC(=O)c2ccccc2N)C(=O)"
        "N[C@@H](C(C)C)C(=O)O[C@@H]1C",
        _declined(),
        marks=_GATE,
    ),
    ("CC(C)CCCCCCCC=O", "9-methyldecanal"),
    # a phase Plan 02 Task 03: tetracyclic flavone dimer. Pre-148 yielded
    # `6,6-dimethyl-2H-pyran` (already a known-bad partial); post-148
    # cascade unblock + decomposition fragment-naming bug yields
    # `2-cycloheptadecylpropan-2-ol` (different partial). a phase / IM-x.x
    # decomposition layer fix territory. Marked xfail; NOT a a phase
    # regression — both old and new are partial outputs.
    pytest.param(
        "CC1(C)C=Cc2c(cc(O)c3c(=O)c4ccc(O[C@@H]5c6c(cc(O)c7c(=O)"
        "c8cccc(O)c8oc67)O[C@H]5C(C)(C)O)c(O)c4oc23)O1",
        "6,6-dimethyl-2H-pyran",  # pre-148 was 2,3-dicyclohexyl-...; both partial
        marks=pytest.mark.xfail(
            strict=True,
            reason="Phase 149 / IM-x.x: decomposition fragment-naming bug for "
                   "tetracyclic flavone dimers ('cycloheptadecyl' in output); "
                   "per 148-01-SUMMARY Risks §2 (Plan-01 carry-forward).",
        ),
    ),
    pytest.param(
        "Nc1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1",
        _ships(
            "4-amino-1-[(2R,3R,4S,5R)-3,4-dihydroxy-5-(hydroxymethyl)oxolan-2-yl]-1,3,5-triazin-"
            "2(1H)-one"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "O=c1c2c(O)cccc2oc2c3c(cc(O)c12)OC1OCCC31",
        _declined(),
        marks=_GATE,
    ),
    # L5 (TRIAGE default-tier-pin-or-decline): the old text is unparseable (an 'oxanyloxyl'
    # glycoside on an 'azacyclotetracosan-2-one'). With the gate off the lactam producer named
    # the ring saturated (every ring double bond lost) and without its amino-sugar glycoside;
    # OPSIN cannot parse that. The PIN tier declines; best-effort names it exactly.
    pytest.param(
        "CC1=C\\C=C\\C(C)=C\\C[C@H](C)NC(=O)/C(CC(C)C)=C/C(C)=C"
        "/C=C/C=C/[C@@](C)(O)[C@@H](O[C@@H]2OC[C@@H](O[C@H]3C"
        "[C@@](C)(O)[C@H](N(C)C)[C@@H](C)O3)[C@H](O)[C@H]2N)"
        "/C=C\\C=C\\1",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "CC(C)[C@@H](NC(N)=O)C(=O)O",
        _ships("(2R)-2-(carbamoylamino)-3-methylbutanoic acid"),
        marks=_GATE,
    ),
    pytest.param(
        "C/C1=C/[C@@H](C)C/C=C\\[C@H]2[C@@H]3O[C@]3(C)[C@@H](C)"
        "[C@H]3C(Cc4c[nH]c5ccccc45)NC(=O)[C@]32C2=N[C@@H](CC2)C1=O",
        _declined(),
        marks=_GATE,
    ),
    # L5 (TRIAGE gateoff-chain-handler-names-fragment): the old text is unparseable. With the
    # gate off the polyfunctional producer named '2-oxobutanoic acid' (C4H6O3 for C9H13NO6S),
    # skipping the S-(2-acetamido-2-carboxyethyl) thioether branch on the claim that the
    # functional-group loop names it; that loop had emitted no prefix. The producer now
    # declines such a branch. The PIN tier builds no verified name; best-effort round-trips.
    pytest.param(
        "CC(=O)N[C@@H](CSCCC(=O)C(=O)O)C(=O)O",
        _tier_contract(),
        marks=_GATE,
    ),
    (
        "O=C(/C=C/c1ccc(Cl)cc1)c1ccccc1",
        "(2E)-3-(4-chlorophenyl)-1-phenylprop-2-en-1-one",
    ),
    pytest.param(
        "CN(C)CCC=C1c2ccccc2COc2ccccc21",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "N=C(N)NCCC[C@H](NC(=O)CNC(=O)[C@@H](N)CCC(=O)O)C(=O)CCl",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "C=C[C@@H]1C(=C)CC[C@H]2[C@H]1C[C@H]1OC(=O)[C@@]3(C)"
        "[C@H](O)CC[C@@]2(C)[C@@]13O",
        _declined_at_default(
            "(1R,2S,6S,7R,9R,12S,13R,16S)-6-ethenyl-13,16-dihydroxy-1,12-dimethyl-5-methylidene-"
            "10-oxatetracyclo[7.6.1.0^2,7.0^12,16]hexadecan-11-one"
        ),
        marks=_GATE,
    ),
    # L4 (TRIAGE ester-glue): the old '4-methyl-5-oxooxolane' names a fragment. The ester
    # producer glued '(2E)-2-methylbut-2-enoate' after an alcohol named with its OH as a
    # 'hydroxy' prefix; OPSIN reads that name as C20H26O8, the input is C20H24O7. Tier
    # contract.
    pytest.param(
        "C=C1C(=O)OC2/C=C(/CO)C(=O)/C=C\\C(C)(O)CC(OC(=O)/C(C)=C/C)C12",
        _tier_contract(),
        marks=_GATE,
    ),
    # "Functional class nomenclature for general nomenclature"
    # (the Blue Book): 'propanal oxime... N-hydroxypropan-1-imine (PIN)'; the
    # oxime is named substitutively, 'N-hydroxy' cited among the prefixes in
    # alphanumerical order. OPSIN full InChIKey exact.
    ("CSCCCCC=NO", "N-hydroxy-5-(methylsulfanyl)pentan-1-imine"),
    ("O=Cc1ccc2ccccc2c1O", "1-hydroxynaphthalene-2-carbaldehyde"),
    pytest.param(
        "C=CC1=C(C)C2=[N]3->[Fe]45<-[N]6=C(C=c7c(CCC(=O)O)c(C)c"
        "([n]74)=C2)[C@]2(CCC(=O)O2)[C@@](C)(O)C6=Cc2c(C=C)c(C)"
        "c([n]25)C=C13",
        # An iron porphyrin is an organometallic, out of scope): declined at every tier.
        _out_of_scope("iron compound (not supported)", "UNSUPPORTED_ELEMENT"),
        marks=_GATE,
    ),
    # L5 (TRIAGE composer-drops-unnameable-organic-substituent): the old text is a wrong
    # molecule ('3-decyloxy-1-hydroxyhydroxydeca-6,8-dien-5-one'). With the gate off the
    # composer skipped the 3-(alkoxy) branch it could not name and shipped
    # '(6E,8E)-1-hydroxydeca-6,8-dien-5-one' (OPSIN: C10H16O2 for the input's C20H32O5). It now
    # voids the candidate for every unnameable branch. The PIN tier declines; best-effort
    # names the whole molecule exactly.
    pytest.param(
        "C/C=C/C=C/C(=O)CC(CCO)OCC[C@H](O)CC(=O)CC/C=C/C",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "CC(=O)N[C@H]1[C@H](OC[C@H]2O[C@@H](O[C@H]3[C@H](O)"
        "[C@@H](O)C(O)O[C@@H]3CO)[C@H](O)[C@@H](O)[C@H]2O)"
        "O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H]"
        "(O[C@H]3O[C@H](CO)[C@H](O)[C@H](O)[C@H]3O)[C@H]2O)"
        "[C@@H]1O",
        _ships(
            "α-D-galactopyranosyl-(1->3)-β-D-galactopyranosyl-(1->4)-2-acetamido-2-deoxy-"
            "β-D-glucopyranosyl-(1->6)-β-D-galactopyranosyl-(1->4)-D-glucopyranose"
        ),
        marks=_GATE,
    ),
    (
        "O=C([O-])[C@H](O)[C@H](O)COP(=O)([O-])[O-]",
        # j7 (TRIAGE g2 G2-C7): the trianion's junior -O-P(=O)(O-)2 is the anionic
        # prefix 'phosphonatooxy' BB:41213, twin of the PIN prefix
        # 'phosphonooxy' BB:36333); OPSIN full-InChIKey exact. The
        # old value was not a parseable name; the engine had also shipped the
        # charge-dropping monoanion '...-4-(phosphonooxy)butanoate' with the gate off.
        "(2R,3R)-2,3-dihydroxy-4-(phosphonatooxy)butanoate",
    ),
    pytest.param(
        "CCCCCCCC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H]"
        "(CCC(=O)O)C(=O)N[C@@H](CC(C)C)C(=O)O)C(C)C",
        _declined_at_default("octanoylvalylphenylalanylglutamylleucine"),
        marks=_GATE,
    ),
    pytest.param(
        "OCCCO",
        _ships("propane-1,3-diol"),
        marks=_GATE,
    ),
    (
        "O=C([O-])[C@@](O)(CO)C(=O)CO",
        "(2R)-2,4-dihydroxy-2-(hydroxymethyl)-3-oxobutanoate",  #: locant-aware prefix merge
    ),
    pytest.param(
        "CCC1CC=C(N2CCCC2)C1=O",
        _ships("5-ethyl-2-(pyrrolidin-1-yl)cyclopent-2-en-1-one"),
        marks=_GATE,
    ),
    pytest.param(
        "Oc1ccc(CC2(O)Oc3cc(O)cc(O)c3C2O)cc1",
        _ships("2-[(4-hydroxyphenyl)methyl]-2,3-dihydro-1-benzofuran-2,3,4,6-tetrol"),
        marks=_GATE,
    ),
    pytest.param(
        "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)"
        "OC[C@H](N)C(=O)O)OC(=O)CCCCCCCCC/C=C\\C/C=C\\CCCCC",
        # Declined at the default tier (NO_VERIFIED_PIN); the strict path's name is pinned, and
        # the best-effort tier gives the same name. Its enclosing marks follow
        # 'Multiple types of enclosing marks' (the Blue Book, nesting order {[({})]}
        #:7446): the propoxy group holds a stereo '(2R)' and two '{' groups, so its natural
        # level-4 mark '(' would stand next to '(2R)'; (:7509) "When the nesting
        # order given in results in consecutive enclosing marks of the same level, the
        # next level of enclosing mark is used" escalates it to '[', and the outer group is
        # '{' -- the Blue Book's own '(3S)-2-[(2S)-2-{[(2S)-...]amino}propanoyl]-...' (:7509ff).
        # Re-derived independently in tests/unit/rules/test_leads_l7_39.py.
        _declined_at_default(
            "O-{[(2R)-2-{[(11Z,14Z)-icosa-11,14-dienoyl]oxy}-3-{[(9Z,12Z,15Z)-octadeca-9,12,15-"
            "trienoyl]oxy}propoxy]hydroxyphosphoryl}-L-serine"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "C[NH2+][C@@H](C)[C@@H](O)c1ccccc1",
        _ships("(1S,2S)-1-hydroxy-N-methyl-1-phenylpropan-2-aminium"),
        marks=_GATE,
    ),
    # fix a performance pass (was 2,5-dithiahexane): (the Blue Book), two
    # heterounits -> substitutive, the analog of "1,2-dimethoxyethane (PIN)" (:27754).
    ("CSCCSC", "1,2-bis(methylsulfanyl)ethane"),
    # L4: the old value is unparseable text. Since 4e4e7cc52 the default tier emits a name only
    # when the strict PIN path built and verified it, and no producer builds a PIN for this
    # phospholipid; best-effort names it and the name round-trips (tier contract).
    pytest.param(
        "CCCCCc1oc(CCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC"
        "[N+](C)(C)C)OC(=O)CCC/C=C\\C[C@H]2[C@@H](O)CC(O)O"
        "[C@@H]2/C=C/[C@@H](O)CCCCC)c(C)c1C",
        _tier_contract(),
        marks=_GATE,
    ),
    # L5 (TRIAGE default-tier-pin-or-decline): the old text '(2R)-2-octyloxypropanoic acid' is
    # a different molecule. With the gate off the polyfunctional producer named '(2R)-propanoic
    # acid', skipping the glucosaminyl ether branch it had no prefix for. It now declines. The
    # PIN tier builds no verified name; best-effort names the ether exactly.
    pytest.param(
        "CC(=O)N[C@H]1C(O)O[C@H](CO)[C@@H](O)[C@@H]1O[C@H](C)"
        "C(=O)O",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "COc1cc(OC)c2c(O)c3c(c(-c4c5cc(OC)cc(OC)c5c(O)c5c(=O)cc(C)"
        "oc45)c2c1)O[C@](C)(O)CC3=O",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "OC[C@H]1O[C@H](OC[C@H]2O[C@H](OC[C@H]3O[C@H](O)[C@H](O)"
        "[C@@H](O)[C@@H]3O)[C@H](O)[C@@H](O)[C@H]2O)[C@H](O)"
        "[C@@H](O)[C@@H]1O",
        _ships("α-D-glucopyranosyl-(1->6)-α-D-galactopyranosyl-(1->6)-α-D-glucopyranose"),
        marks=_GATE,
    ),
    pytest.param(
        "N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CS)C(=O)N[C@@H](CS)C(=O)O",
        _ships(
            "(2R)-2-{(2R)-2-[(2S)-2-amino-3-phenylpropanamido]-3-sulfanylpropanamido}-3-sulfanylpropanoic acid"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "COc1cc(OC)c2c(c1CC=C(C)C)O[C@H](c1ccccc1)CC2",
        _ships("(2S)-5,7-dimethoxy-8-(3-methylbut-2-en-1-yl)-2-phenyl-3,4-dihydro-2H-1-benzopyran"),
        marks=_GATE,
    ),
    # L5 (TRIAGE gateoff-chain-handler-names-fragment): the old text is unparseable. With the
    # gate off the polyfunctional producer named 'propanoic acid' (C3H6O2 for the input's
    # C56H95N3O17), skipping the macrolide O-substituent of the malonate half-ester. It now
    # declines a branch no functional-group prefix names. The PIN tier declines; best-effort
    # round-trips.
    pytest.param(
        "CN=C(N)NCCC/C=C/CCC[C@H](C)[C@H]1OC(=O)/C(C)=C\\C=C/"
        "[C@H](C)[C@H](O)C[C@H](O)[C@H](C)[C@@H](O)CC[C@@H](C)"
        "[C@H](O)C[C@@]2(O)O[C@H](C[C@H](O)C[C@H](OC(=O)CC(=O)O)"
        "C[C@@H](O)C[C@H](O)/C(C)=C\\C=C/[C@H]1C)C[C@@H](O)"
        "[C@@H]2O",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "NC(CCC(=O)NC(CSC(CC=O)c1ccccc1O)C(=O)NCC(=O)O)C(=O)O",
        # Declined at the default tier (NO_VERIFIED_PIN); the strict path's name is pinned (the
        # best-effort tier names it differently; that name must round-trip). Its enclosing marks
        # follow (the Blue Book, order:7446): the deepest child of the propanoyl
        # group is the '{...}' (level 3), so its mark is the level-4 '(' (the first child
        # '(4-amino-4-carboxybutanamido)' is not adjacent to it, so:7509 does not
        # escalate), and that child keeps its own level as a sibling does in '4-(6-{2-[(3-
        # methylphenyl)methylidene]hydrazin-1-yl}-2-[2-(pyridin-2-yl)ethoxy]pyrimidin-4-yl)
        # morpholine (PIN)' (:19441, 'General methodology':19420). Re-derived
        # independently in tests/unit/rules/test_leads_l7_39.py.
        _declined_at_default(
            "N-(2-(4-amino-4-carboxybutanamido)-3-{[1-(2-hydroxyphenyl)-3-oxopropyl]sulfanyl}"
            "propanoyl)glycine",
            best_effort_same=False,
        ),
        marks=_GATE,
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\CCCCCC[C@@H](OO)C(=O)[O-]",
        "(2R,9Z,12Z,15Z)-2-hydroperoxyoctadeca-9,12,15-trienoate",
    ),
    pytest.param(
        "Cc1c(Cl)c(O)cc2oc(=O)c3c(O)cc(O)cc3c12",
        _ships("2-chloro-3,7,9-trihydroxy-1-methyl-6H-dibenzo[b,d]pyran-6-one"),
        marks=_GATE,
    ),
    pytest.param(
        "CN1C(=O)[C@]23SSS[C@@]1(CO)C(=O)N2[C@H]1Nc2ccccc2"
        "[C@@]1(c1c[nH]c2ccccc12)[C@@H]3O",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "C[C@H](O)/C=C1\\C[C@H](O)[C@]23C[C@H]2C(C)(C)O[C@]3(O)C1=O",
        _declined_at_default(
            "(1S,3R,6S,8E,10S)-6,10-dihydroxy-8-[(2S)-2-hydroxypropylidene]-4,4-dimethyl-"
            "5-oxatricyclo[4.4.0.0^1,3]decan-7-one"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "C=C1[C@@H](O)CC[C@]2(C)C3=C(CC[C@@H]12)[C@]1(O)[C@@H](O)"
        "C[C@H]([C@H](C)CC[C@H](CC)C(C)C)[C@@]1(C)C[C@@H]3O",
        _declined(),
        marks=_GATE,
    ),
    # (the Blue Book) '=S: -thione and sulfanylidene';
    # (:29561) C=O > C=S > C=Se > C=Te are ketonic suffixes, and ketones, pseudoketones
    # and heterones (class 16,:18189) outrank amines (class 19): the ring thione
    # is the suffix, the amine a prefix. Indicated hydrogen takes the lowest locant
    # ('2(1H)', not '2(3H)'), which puts the amino group at C-6. OPSIN full InChIKey exact.
    ("Nc1[nH]c(=S)ncc1F", "6-amino-5-fluoropyrimidine-2(1H)-thione"),
    # L5 (TRIAGE composer-drops-unnameable-organic-substituent): the old text is a wrong
    # molecule (OPSIN: C26H40N4O7 for the input's C27H42N4O7). With the gate off the acylamino
    # prefix builder spelled the N-methylated leucine acyl as '...propanoylamino', the N-methyl
    # lost.
    # It now declines an acyl whose N carries a further substituent. The PIN tier declines;
    # best-effort round-trips.
    pytest.param(
        "CC[C@@H](C)[C@H](NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](C)"
        "NC(=O)[C@H](CCO)NC(=O)c1ccccc1)C(=O)O",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "CC(=O)N[C@@H]1[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@H](O)"
        "[C@H](O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@@H]4O[C@H](CO)"
        "[C@H](O)[C@H](O)[C@H]4O)[C@H]3NC(C)=O)[C@H]2O)"
        "[C@@H](CO)O[C@H]1O",
        _ships(
            "β-D-galactopyranosyl-(1->3)-2-acetamido-2-deoxy-α-D-glucopyranosyl-(1->3)-β-"
            "D-galactopyranosyl-(1->4)-2-acetamido-2-deoxy-β-D-glucopyranose"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "CN1CCCN=C1/C=C/c1cccs1",
        _ships("1-methyl-2-[(E)-2-(thiophen-2-yl)ethenyl]-1,4,5,6-tetrahydropyrimidine"),
        marks=_GATE,
    ),
    (
        "CC/C=C/CCCC(=O)CCCCCC(=O)O",
        "(11E)-7-oxotetradec-11-enoic acid",
    ),
    pytest.param(
        "C=C/C(C)=C/[C@]1(C)SC(=O)C(CC)=C1O",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "N[C@H](C=O)Cc1cnc[nH]1",
        _ships("(2S)-2-amino-3-(1H-imidazol-5-yl)propanal"),
        marks=_GATE,
    ),
    pytest.param(
        "O=c1cc(-c2ccc(O)cc2)oc2cc(O)c(Cl)c(O)c12",
        _ships("6-chloro-5,7-dihydroxy-2-(4-hydroxyphenyl)-4H-1-benzopyran-4-one"),
        marks=_GATE,
    ),
    (
        "CCCCCCC[C@@H](O)[C@H](O)CC#CC#C[C@@H](O)CC",
        "(3S,9R,10R)-heptadeca-4,6-diyne-3,9,10-triol",
    ),
    pytest.param(
        "CC1CCC/C=C\\C=C\\C(O)CC(O)C/C=C\\C=C\\C(O)C/C=C/C=C\\C(=O)O1",
        # (heading 'General methodology', the Blue Book),:16497:
        # "For euphonic reasons, when the endings 'ene' and 'yne' are preceded by a multiplying
        # prefix and a locant the letter 'a' is inserted." (cf. '1-oxacycloundeca-2,4,6,8,10-
        # pentaene (PIN)',:8498). The macrolactone namer used to ship '...oxacyclotetracos-
        # 3,5,9,11,17,19-hexaen-2-one' (OPSIN read it back exactly, but it lacks the 'a'); the
        # name below is the Blue Book spelling, also read back exactly (leads L7 / 38).
        _ships(
            "(3Z,5E,9E,11Z,17E,19Z)-8,14,16-trihydroxy-24-methyl-1-oxacyclotetracosa-3,5,9,"
            "11,17,19-hexaen-2-one"
        ),
        marks=_GATE,
    ),
    # L5 (TRIAGE gateoff-chain-handler-names-fragment): the old text is a wrong molecule.
    # With the gate off the polyfunctional producer named a C20 chain with a 'hexol' suffix
    # (OPSIN: C40H68O8 for the input's C29H51O12P): the six OH are the inositol's, on no atom
    # of that chain. It now declines, the Blue Book). The PIN tier declines;
    # best-effort names the phosphatidylinositol exactly.
    pytest.param(
        "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)OC[C@@H](O)COP(=O)(O)"
        "OC1C(O)C(O)C(O)[C@@H](O)C1O",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "CCCCCCCC(O)CC(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CCC(=O)O)"
        "C(=O)N[C@H]1C(=O)N[C@H](C(C)C)C(=O)N[C@@H](CC(C)C)"
        "C(=O)N[C@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CO)"
        "C(=O)N[C@@H]([C@@H](C)CC)C(=O)OC1C",
        _declined(),
        marks=_GATE,
    ),
    # L5 (TRIAGE default-tier-pin-or-decline): the old text is unparseable. With the gate off
    # the cage producer named the glycosyloxy / ester branches by their carbon counts
    # ('15-docosyl-...', OPSIN C61H102O7 for the input's C61H86O22). A branch that is not an
    # unbranched saturated alkyl now declines the cage. The PIN tier declines; best-effort
    # names it and the name round-trips.
    pytest.param(
        "CC[C@@H]1C[C@@]23OC(=O)C(=C2O)OC(=O)[C@]2(C)[C@H](CCCC"
        "[C@]3(C)C=C1C(=O)O)C(C)=C[C@@H]1[C@@H](OC3OC(C)C(OC(=O)"
        "c4c(C)cccc4OC)C(OC4CC(O)C(OC)C(C)O4)C3O)[C@@H](OC3OC(C)"
        "C(OC)C(C)(O)C3OC)CC[C@H]12",
        _tier_contract(),
        marks=_GATE,
    ),
    (
        "CC(C)=CCC[C@@H](C(=O)O)[C@H]1C(=O)C[C@@]2(C)C3=C(CC"
        "[C@]12C)[C@@]1(C)CCC(=O)C(C)(C)[C@@H]1[C@@H](O)C3",
        # j7 (TRIAGE g2 G2-C6): the terminal -COOH is the '-21-oic acid' suffix,
        # the Blue Book), the ketones 'oxo' prefixes; was '6,21-dihydroxy...
        # -3,16,21-trione'. name, labelled best_effort; OPSIN full-InChIKey exact.
        "(5R,6S,10S,13R,14R,17R,20R)-6-hydroxy-4,4,14-trimethyl-3,16-dioxocholesta-8,24-dien-21-oic acid",
    ),
    (
        # Suite fix j6 (TRIAGE g2 G2-C8): the PIN tier abstained (a branched
        # unsaturated substituent was declined). (the Blue Book):
        # the parent carries the maximum number of principal characteristic
        # groups -- benzene-1,4-diol (2 OH), not the butenol (1 OH); the old
        # value was OPSIN-exact but not the PIN. OPSIN 2.9.0: exact.
        "CC(C)(O)/C=C/c1cc(O)ccc1O",
        "2-[(1E)-3-hydroxy-3-methylbut-1-en-1-yl]benzene-1,4-diol",
    ),
    (
        "CC(C)=CCC(C)/C(C)=C/CO",
        "(2E)-3,4,7-trimethylocta-2,6-dien-1-ol",
    ),
    (
        "CCCCCC=CCC=CCCCCCCCC(=O)OCC(O)COP(=O)(O)OCCN",
        # (the Blue Book): esters (class 9,:18182) outrank alcohols (class 17,
        #:18190), so the ester is the parent, cited as 'alkyl alkanoate';
        # the phosphoric acid diester and the free hydroxyl are prefixes of the 'yl'
        # word (cf. Phosphatidylethanolamine,:55172, example name:55180). OPSIN
        # full InChIKey exact.
        "3-{[(2-aminoethoxy)hydroxyphosphoryl]oxy}-2-hydroxypropyl octadeca-9,12-dienoate",
    ),
    pytest.param(
        "COc1cc(OC2OC(C(=O)O)C(O)C(O)C2O)c(C2CC(=O)c3ccc(O)cc3O2)"
        "c(O)c1CC=C(C)C",
        # fix-all merge: the PIN path now builds and verifies this name (no stereo in the
        # input, so no carbohydrate name); OPSIN 2.9.0 full InChIKey exact.
        _ships("3,4,5-trihydroxy-6-[3-hydroxy-2-(7-hydroxy-4-oxo-3,4-dihydro-2H-1-"
               "benzopyran-2-yl)-5-methoxy-4-(3-methylbut-2-en-1-yl)phenoxy]oxane-2-"
               "carboxylic acid"),
        marks=_GATE,
    ),
    # L4 (TRIAGE ester-glue): the old text is unparseable. The ester producer glued
    # 'acetate' after a name with a 7-'hydroxy' (OPSIN reads it as C28H38ClNO7, the input is
    # C28H36ClNO6). Tier contract.
    pytest.param(
        "CC[C@H](C)C=C(C)C=CC1=CC2=C(Cl)C(=O)[C@@](C)(OC(C)=O)"
        "C(=O)C2=CN1C(CC(C)C)C(=O)OC",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "COc1ccc2c(c1OC)C(Cc1ccc(O)cc1)[N+](C)(C)CC2",
        _declined(),
        marks=_GATE,
    ),
    (
        "CCC/C=C/C/C=C/C/C=C/CCCCC(=O)O",
        "(6E,9E,12E)-hexadeca-6,9,12-trienoic acid",
    ),
    pytest.param(
        "CC1=CC2=C(C=O)C(=O)C(C)(O)C(O)C2=CO1",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "COc1cc(/C=C/C(=O)O[C@H]2[C@H](O)C[C@](O)(C(=O)O)C[C@H]2O)"
        "ccc1O",
        # OPSIN 2.9.0 cannot assign this name's CIP labels, so the default tier ships it on the
        # constitution-only comparison; the stereo is checked against RDKit's labeller.
        _constitution_only(
            "(1S,3R,4S,5R)-1,3,5-trihydroxy-4-{[(2E)-3-(4-hydroxy-3-methoxyphenyl)prop-2-"
            "enoyl]oxy}cyclohexane-1-carboxylic acid"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "O=C1N=C([O-])c2ccccc21.[K+]",
        _declined("UNSUPPORTED_ELEMENT"),
        marks=_GATE,
    ),
    pytest.param(
        "CC1=C(O)C(=O)C([C@@]2(C)CCCC2(C)C)=C(O)C1=O",
        _ships(
            "2,5-dihydroxy-3-methyl-6-[(1S)-1,2,2-trimethylcyclopentyl]cyclohexa-2,5-diene-"
            "1,4-dione"
        ),
        marks=_GATE,
    ),
    # L4 (TRIAGE ester-glue): the old text is a wrong molecule. The ester producer glued
    # '2,3-dihydroxybenzoate' after an alcohol named without its linking oxygen (OPSIN reads it
    # as C26H32N4O12, the input is C26H30N4O11). It now builds the group word from structure.
    # Tier contract.
    pytest.param(
        "CC(=O)N(O)CCCCNC(=O)[C@H](COC(=O)c1cccc(O)c1O)NC(=O)"
        "[C@@H]1COC(c2cccc(O)c2O)=N1",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "CC(C)C1=C[C@@]23CC[C@H]4C(C)(C)CCC[C@]4(C(=O)O2)C3=CC1=O",
        _ships(
            "(4aR,8aR,10aS)-1,1-dimethyl-7-(propan-2-yl)-1,3,4,9,10,10a-hexahydro-2H,6H-8a,"
            "4a-(epoxymethano)phenanthrene-6,12-dione"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "COC1=C(N[C@H](C(=O)O)[C@@H](C)O[C@@H]2O[C@H](CO)[C@H](O)"
        "[C@H](O)[C@H]2O)C[C@](O)(CO)CC1=NCC(=O)O",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "CC(=O)CCC1=C(C)C[C@@]2(CC1=O)C(=O)[C@@H]1C[C@@](O)(CO1)C2=O",
        # The ring ketones are the principal characteristic group class 16,
        # the Blue Book, above the hydroxy compounds of class 17), so the spiro
        # parent takes the suffix '-2,4,5'-trione' and the hydroxy, methyl and
        # 3-oxobutyl groups are prefixes. OPSIN full InChIKey exact (this spelling, the
        # numbering below is the one the component-name spiro core gives).
        "(1R,3R,5S)-1-hydroxy-3'-methyl-4'-(3-oxobutyl)-6-oxaspiro"
        "[bicyclo[3.2.1]octane-3,1'-cyclohex-3-ene]-2,4,5'-trione",
        marks=pytest.mark.xfail(strict=True, reason=(
            "spiro-vonbaeyer-component parent names its ring ketones as 'oxo' prefixes: "
            "the component-name spiro core (rules/spiro.py _name_spiro_vonbaeyer_core, "
            "numbering 'not re-optimised for lowest substituent locants') has no suffix "
            "form and the general engine's spiro analysis declines a von Baeyer "
            "component; the numbering of the suffix form (lowest locants to the "
            "ketones, P-31.1.4.2.4) is ASSUMED -- see "
            ".planning/preexisting-triage/TRIAGE-2026-10-09.md")),
    ),
    # L4: the ester of an enol, named as functional class: "Definitions"
    # (the Blue Book) "... formally derived from an organic oxoacid... and an alcohol,
    # phenol, heterol, or enol", (:31663) "All preferred IUPAC names for esters are
    # named by functional class nomenclature" ('ethyl acetate'), so the first word is the group
    # name. The ketone ranks below the ester, so it is 'dioxo' inside the group. The old
    # value is a wrong molecule and the raw name '2-dodecyl-3-hydroxynaphthalene-1,4-dione
    # acetate' glued the alcohol's own name in front of 'acetate' (OPSIN reads two molecules).
    # OPSIN 2.9.0 full-InChIKey exact; pin_verified with the gate on.
    pytest.param(
        "CCCCCCCCCCCCC1=C(OC(C)=O)C(=O)c2ccccc2C1=O",
        _ships("3-dodecyl-1,4-dioxo-1,4-dihydronaphthalen-2-yl acetate"),
        marks=_GATE,
    ),
    pytest.param(
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O.O",
        _ships("5'-adenylic acid—water (1/1)"),
        marks=_GATE,
    ),
    pytest.param(
        "COc1cc2c(c(O)c1C/C=C(\\C)CCC=C(C)C)CN(CCc1c[nH]c3ccccc13)C2=O",
        _ships(
            "5-[(2E)-3,7-dimethylocta-2,6-dien-1-yl]-4-hydroxy-2-[2-(1H-indol-3-yl)ethyl]"
            "-6-methoxy-2,3-dihydro-1H-isoindol-1-one"
        ),
        marks=_GATE,
    ),
    # L5 (TRIAGE chain-handler-names-fragment): the old text is unparseable ('adenine...
    # imidazolylpropanoate'). With the gate off the polyfunctional producer named the histidine
    # alcohol '(2S)-2-amino-3-(1H-imidazol-4-yl)propanol' (OPSIN: C6H11N3O for the input's
    # C16H21N8O8P): it cited an '-ol' suffix for an OH that is on no atom of its chain. It now
    # declines, the Blue Book, the suffix groups must sit on the parent). The
    # PIN tier builds no verified name; best-effort names it and the name round-trips.
    pytest.param(
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](OC(=O)"
        "[C@@H](N)Cc2c[nH]cn2)[C@H]1O",
        _tier_contract(),
        marks=_GATE,
    ),
    pytest.param(
        "NC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](N)C[C@@H]2N)"
        "[C@H](O)[C@@H](O)[C@@H]1O",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "CSCC[C@H](NC(=O)[C@H](CO)NC(=O)[C@@H](N)CCCCN)C(=O)O",
        _ships(
            "(2S)-2-{(2S)-2-[(2S)-2,6-diaminohexanamido]-3-hydroxypropanamido}-4-(methylsulfanyl)"
            "butanoic acid"
        ),
        marks=_GATE,
    ),
    pytest.param(
        "CSCC[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](Cc1cnc[nH]1)"
        "C(=O)O",
        _declined_at_default("methionylasparaginylhistidine"),
        marks=_GATE,
    ),
    pytest.param(
        "CC(C)=CCc1c(O)ccc(C(=O)C2C(c3c(O)cc(/C=C/c4cc(O)c(O)cc4O)"
        "cc3O)C=C(C)CC2c2ccc(OC3OC(C(=O)O)C(O)C(O)C3O)cc2O)c1O",
        _declined(),
        marks=_GATE,
    ),
    pytest.param(
        "COC(=O)c1ccccc1OC1OC(COC2OC(C)C(O)C(O)C2O)C(O)C(O)C1O",
        _declined(),
        marks=_GATE,
    ),
]


class TestCIBenchmark:
    """CI benchmark: 100 ChEBI compounds for regression detection.

    These tests verify deterministic IUPAC name output for a representative
    sample of real-world compounds from the ChEBI database. Failures indicate
    a naming regression that must be investigated before merge.

    Sampling: random.Random(123).sample(111823_unique_smiles, 500)[:100]
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected_name",
        CI_BENCHMARK,
        ids=[f"ci-{i:03d}" for i in range(len(CI_BENCHMARK))],
    )
    def test_ci_benchmark(self, smiles, expected_name):
        """Verify naming output matches pre-computed expected name."""
        if isinstance(expected_name, _TierRow):
            expected_name.check(smiles)
            return
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for {smiles}"
        assert isinstance(name, str), f"Expected string, got {type(name)}"
        assert len(name) > 0, f"Empty name for {smiles}"
        assert name == expected_name, (
            f"CI REGRESSION: {smiles}\n"
            f"  Expected: {expected_name}\n"
            f"  Got:      {name}"
        )
