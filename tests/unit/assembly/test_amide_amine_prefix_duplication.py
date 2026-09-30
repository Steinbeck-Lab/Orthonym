"""Task I (second producer): one nitrogen must never be spelled twice.

`CNCC(=O)N` is 2-(methylamino)acetamide. The amide assembly path emitted
`2-amino-2-(methylamino)acetamide` -- a DIFFERENT MOLECULE with three nitrogens
instead of two. The single secondary-amine nitrogen was spelled twice at one locant:

  * the substituent path named the branch correctly, `(methylamino)`;
  * the functional-group prefix loop *also* emitted a bare `amino`, because
    `seniority.get_prefix` maps `secondary_amine`/`tertiary_amine` to the bare
    string `"amino"`.

Bare `amino` is not a legal spelling for a substituted nitrogen. "The
prefix 'amino'": *"Preferred IUPAC names for prefixes corresponding to -NHR,
-NRR', or -NR2 are formed by prefixing the names of the groups R and R' to the
prefix 'amino'"* (PIN example `4,4-bis(methylamino)butanoic acid`). So the bare
prefix both duplicates the atom and silently drops R.

The de-duplication had existed only as `BRANCH_HANDLED_FGS`, which covers
`primary_amine` and only branches of <=3 carbons -- so `CCCCNCC(=O)N`
(butylamino, 4 carbons) duplicated as well. The fix keys on the structural fact
instead: a nitrogen that ATTACHES a substituent branch to the parent is
necessarily spelled by that branch's own name.
"""

import pytest
from rdkit import Chem

import orthonym.namer as _namer
from orthonym import Orthonym


@pytest.fixture(scope="module")
def namer():
    # The defect lives with the validity gate OFF: with it on, merely
    # suppresses the wrong name. Judge the generator, not the gate. The flag is
    # put back when the module ends, so a later module is not named with the gate
    # off.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_namer, "_DISABLE_VALIDITY_GATE", True)
        yield Orthonym(style="pin")


def _name(namer, smiles):
    r = namer.name_tiered(smiles)
    return r.get("name") if isinstance(r, dict) else r


def _inchikey(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CNCC(=O)N", "2-(methylamino)acetamide"),
        ("CCNCC(=O)N", "2-(ethylamino)acetamide"),
        ("CCCCNCC(=O)N", "2-(butylamino)acetamide"),      # 4-carbon branch
        ("CN(C)CC(=O)N", "2-(dimethylamino)acetamide"),   # tertiary N
        ("CNCCC(=O)N", "3-(methylamino)propanamide"),     # beta, not alpha-specific
    ],
)
def test_substituted_amine_nitrogen_spelled_once(namer, smiles, expected):
    assert _name(namer, smiles) == expected


@pytest.mark.parametrize(
    "smiles",
    ["CNCC(=O)N", "CCNCC(=O)N", "CCCCNCC(=O)N", "CN(C)CC(=O)N", "CNCCC(=O)N"],
)
def test_no_spurious_bare_amino_prefix(namer, smiles):
    """The fabricated `amino` prefix invented a nitrogen that is not there."""
    name = _name(namer, smiles)
    assert name is not None
    # Enclosing marks hide the duplication from a raw substring test, so strip
    # them first: pre-fix '2-amino-2-(methylamino)acetamide' flattens to
    # '2-amino-2-methylaminoacetamide' and counts TWO. Exactly one nitrogen is
    # substituted in every row here, so exactly one 'amino' token is correct.
    flat = name.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    assert flat.count("amino") == 1, f"nitrogen spelled twice: {name!r}"
    # a bare 'amino' prefix would appear as '<locant>-amino-'
    assert "-amino-" not in name, f"bare amino prefix duplicates the N: {name!r}"


# --- regression anchors: genuine primary amines must KEEP their amino prefix ---

@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("NCC(=O)N", "2-aminoacetamide"),      # glycinamide: real -NH2
        ("CC(=O)N", "acetamide"),
        ("CCC(=O)N", "propanamide"),
        ("OCC(=O)N", "2-hydroxyethanamide"),
        ("CSCC(=O)N", "2-(methylsulfanyl)ethanamide"),
    ],
)
def test_primary_amine_and_plain_amides_unchanged(namer, smiles, expected):
    assert _name(namer, smiles) == expected


def test_nitrogen_count_matches_structure(namer):
    """The emitted name must not invent or drop a nitrogen (atom coverage)."""
    for smiles in ("CNCC(=O)N", "CCCCNCC(=O)N", "CN(C)CC(=O)N"):
        name = _name(namer, smiles)
        assert name is not None
        # exactly one 'amino' token (the substituent), never two
        assert name.count("amino") == 1, f"{smiles} -> {name!r}"
