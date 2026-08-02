"""N-substituted sulfonamides — P-66.1.1.3.1.1 producer-level tests.

Blue Book authority
-------------------
**P-66.1.1.2** "Sulfonamides, sulfinamides, and related selenium and tellurium
amides" (``BlueBookV2/BlueBookV2.md:32746``):

    "Sulfonamides, sulfinamides, and the analogous selenium and tellurium
    amides are named substitutively using the following suffixes:
    -SO2-NH2 sulfonamide (preselected suffix) ... These suffixes may be
    assigned to any position of a parent hydride."

Worked ``(PIN)``: ``CH3-SO2-NH2 methanesulfonamide (PIN)`` (``:32752``).

**P-66.1.1.3.1** "*N*-Substitution", subsection **P-66.1.1.3.1.1**
(``BlueBookV2.md:32774``) — the decisive sentence:

    "Substituted primary amides, with general structures such as R-CO-NHR' and
    R-CO-NR'R'', **and the corresponding amides derived from chalcogen acids**
    are named by citing the substituents R' and R'' as prefixes preceded by the
    locant *N* when one amide group is present."

A sulfonamide is an amide derived from a chalcogen (sulfur) acid, so the italic
``N-`` locant is REQUIRED here. Confirmed by the Blue Book's own ``(PIN)``
sulfonamide examples carrying that locant:

* ``N*3-ethyl-N^1-methylnaphthalene-1,3-disulfonamide (PIN)`` (``:32805``)
* ``3-chloro-N-(2-chlorophenyl)naphthalene-2-sulfonamide (PIN)`` (``:32881``)
  — listed AGAINST the non-PIN ``2',3-dichloronaphthalene-2-sulfonanilide``,
  which is why the anilide contraction must never be emitted.
* ``N-carbamoylbenzenesulfonamide (PIN)`` (``:33362``)
* ``N-hydroxymethanesulfonamide (PIN)`` (``:38338``)

These tests assert at the PRODUCER (``n_substituted_sulfonamide_name``), never
through ``name_compound`` — the OPSIN self-consistency gate suppresses a wrong
name before any assertion could see it, which is how two earlier mutations
survived a green suite.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.rules.sulfonamides import n_substituted_sulfonamide_name


def _name(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles!r}"
    return n_substituted_sulfonamide_name(mol)


# --------------------------------------------------------------------------
# Mono-N-substitution, methanesulfonamide-type (chain) parent
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("CS(=O)(=O)NC",         "N-methylmethanesulfonamide"),
    ("CS(=O)(=O)NCC",        "N-ethylmethanesulfonamide"),
    ("CS(=O)(=O)NC(C)C",     "N-(propan-2-yl)methanesulfonamide"),
    ("CS(=O)(=O)NC(C)(C)C",  "N-tert-butylmethanesulfonamide"),
    ("CCS(=O)(=O)NC",        "N-methylethanesulfonamide"),
])
def test_mono_n_substituted_chain_parent(smiles, expected):
    assert _name(smiles) == expected


# --------------------------------------------------------------------------
# Aryl / cycloalkyl / benzyl N-substituents
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("CS(=O)(=O)Nc1ccccc1",   "N-phenylmethanesulfonamide"),
    ("CS(=O)(=O)NC1CCCCC1",   "N-cyclohexylmethanesulfonamide"),
    ("CS(=O)(=O)NCc1ccccc1",  "N-benzylmethanesulfonamide"),
])
def test_aryl_and_cycloalkyl_n_substituents(smiles, expected):
    assert _name(smiles) == expected


def test_n_phenyl_is_not_contracted_to_an_anilide():
    """P-66.1.1.3.1.1 / :32881 — the sulfonanilide form is explicitly non-PIN."""
    assert "anilide" not in _name("CS(=O)(=O)Nc1ccccc1")


# --------------------------------------------------------------------------
# benzenesulfonamide-type (ring) parent
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("CNS(=O)(=O)c1ccccc1",   "N-methylbenzenesulfonamide"),
    ("CCNS(=O)(=O)c1ccccc1",  "N-ethylbenzenesulfonamide"),
])
def test_ring_parent_mono_n_substituted(smiles, expected):
    assert _name(smiles) == expected


# --------------------------------------------------------------------------
# N,N-disubstitution (P-66.1.1.3.1.1 "R-CO-NR'R''")
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("CS(=O)(=O)N(C)C",        "N,N-dimethylmethanesulfonamide"),
    ("CN(C)S(=O)(=O)c1ccccc1", "N,N-dimethylbenzenesulfonamide"),
])
def test_symmetric_n_n_disubstituted(smiles, expected):
    assert _name(smiles) == expected


def test_unsymmetric_n_n_disubstituted_is_alphabetised():
    """Two different N-substituents are cited alphabetically, each with its
    own italic N locant (P-66.1.1.3.1.1)."""
    assert _name("CS(=O)(=O)N(C)CC") == "N-ethyl-N-methylmethanesulfonamide"


# --------------------------------------------------------------------------
# Fail-closed boundary — never emit a wrong constitution
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,why", [
    ("CS(=O)(=O)N",        "primary sulfonamide: not this handler's class"),
    ("NS(=O)(=O)c1ccccc1", "primary sulfonamide: not this handler's class"),
    ("CS(=O)(=O)N1CCCCC1", "N is a RING atom -- PIN is 1-(methanesulfonyl)piperidine"),
    ("CS(=O)(=O)N1CCOCC1", "N is a RING atom"),
    ("CS(=O)NC",           "sulfINamide: no FG perception exists, separate class"),
    ("CC(=O)NC",           "carboxamide, not a sulfonamide"),
    ("NS(=O)(=O)N",        "sulfuric diamide: both N unsubstituted"),
    # -- one mutation-test target each, so each guard has a witness ---------
    ("CNS(=O)(=O)c1ccc(cc1)S(=O)(=O)NC",
     "DIsulfonamide: needs the superscripted N^1/N^3 locants of "
     "P-66.1.1.3.1.1, which are not built here"),
    ("CNS(=O)(=O)N",  "sulfamide: the excised parent is 'sulfuric diamide', "
                      "NOT a sulfonamide, so prefixing 'N-methyl' would name "
                      "a different class"),
    ("CNS(=O)(=O)NC", "N,N'-disubstituted sulfamide: same, two amide N"),
    ("O=S1(=O)CCCN1", "cyclic sulfonamide (sultam): the branch loops back to S"),
    ("O=S1(=O)CCCCN1", "cyclic sulfonamide (sultam)"),
    ("CS(=O)(=O)NO",  "N-hydroxy: a heteroatom N-substituent is P-66.1.1.3.2 "
                      "territory, a different construction"),
    ("CS(=O)(=O)NN",  "N-amino: heteroatom N-substituent"),
])
def test_fails_closed_outside_the_class(smiles, why):
    assert _name(smiles) is None, why


@pytest.mark.parametrize("smiles", [
    "Cc1ccc(cc1)S(=O)(=O)NC",   # 4-methyl on the ring + N-methyl
    "CNS(=O)(=O)c1ccc(O)cc1",   # 4-hydroxy on the ring + N-methyl
    "CNS(=O)(=O)c1ccc(Cl)cc1",  # 4-chloro on the ring + N-methyl
    "CC(C)S(=O)(=O)NC",         # branched chain parent
])
def test_substituted_parent_hydride_fails_closed(smiles):
    """P-66.1.1.3.1.1 / BB:32879 ``N,4-dimethyl-N-(3-methylphenyl)benzamide``:
    when the parent is itself substituted the N and numerical locants form ONE
    ordered prefix list. This module prefixes a delegated parent name and cannot
    merge them, so it must refuse rather than emit ``N-methyl4-methyl...``.

    Regression guard: that malformed string ROUND-TRIPPED through OPSIN to the
    correct InChIKey, so no round-trip test can catch it.
    """
    assert _name(smiles) is None


def test_never_emits_a_concatenated_locant_run():
    """The specific malformed shape observed before the guard existed."""
    out = _name("Cc1ccc(cc1)S(=O)(=O)NC")
    assert out is None or "methyl4" not in out


def test_never_drops_the_n_substituent_carbon():
    """The v29 defect this class fixes: the producer used to emit the parent
    name `methanesulfonamide` for CS(=O)(=O)NC, silently dropping a carbon."""
    assert _name("CS(=O)(=O)NC") != "methanesulfonamide"
