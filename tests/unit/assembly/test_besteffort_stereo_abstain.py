"""Best-effort must ABSTAIN a stereo-INCOMPLETE name, never ship it stripped.

User directive 2026-08-31 (accurate-or-abstain, round-trip metric): a name that
omits stereochemistry the input asserts does not round-trip to the exact input
stereoisomer -- under full standard InChIKey it is a FAIL, not a valid superset.
So the best-effort tier (``general_fallback_unverified``) must decline it exactly
like the complete tier, rather than emit a constitution-only stripped form.

Governing rule for the emission it REPLACES: P-91.2.1 (a name must specify every
stereogenic unit to denote one stereoisomer). The old P-91.2.2 "omit descriptors"
ship path is retained only under ORTHONYM_BE_STRIP_STEREO=1 (measurement).
"""
import os
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound, errors

BE = dict(general_fallback=True, general_fallback_unverified=True,
          allow_aromatic_general=True)


def _full_key(s):
    m = Chem.MolFromSmiles(s)
    return inchi.MolToInchiKey(m) if m else None


def _emits(smiles):
    n = name_compound(smiles, **BE)
    return n if (n and not errors.is_failure_name(n)) else None


# Inputs whose ONLY OST name would strip stereo the input asserts (OPSIN cannot
# express the pseudo-asymmetric / the general engine dropped the E/Z).
STEREO_STRIP_ABSTAINERS = [
    "CN1[C@@H]2CC[C@H]1C[C@H](O)C2",              # tropan-3-ol: PIN uses '3s' (OPSIN-unparseable)
    "O=C([O-])C/C(=C/[PH](=O)[O-])C(=O)[O-]",     # general-engine drops the C=C E/Z
]


# The exact stereo-STRIPPED forms the floor used to ship (wrong molecule under
# full InChIKey). These must NEVER be emitted again.
STRIPPED_FORMS = {
    "CN1[C@@H]2CC[C@H]1C[C@H](O)C2":
        "3-hydroxy-8-(methan-1-yl)-8-azabicyclo[3.2.1]octane",
}


@pytest.mark.parametrize("smiles", STEREO_STRIP_ABSTAINERS)
def test_besteffort_never_ships_stereo_stripped(smiles):
    """The contract: no WRONG-molecule reparse. best-effort must not emit a
    stereo-STRIPPED name that OPSIN reparses to a DIFFERENT full InChIKey. It may
    emit an accurate name that is merely OPSIN-unparseable (e.g. the pseudo-
    asymmetric PIN), or abstain -- both are accurate-or-abstain compliant."""
    from orthonym.namer import _validity_gate_name_to_smiles as n2s
    n = _emits(smiles)
    if n is None:
        return  # abstained -- fine
    # must NOT be the known stripped form
    assert n != STRIPPED_FORMS.get(smiles), (
        "best-effort shipped the stereo-STRIPPED form %r for %s" % (n, smiles))
    osm = n2s(n)
    if osm:  # OPSIN parsed it -> it MUST round-trip the full InChIKey (no wrong)
        assert _full_key(osm) == _full_key(smiles), (
            "best-effort shipped a wrong-molecule/stereo-stripped name %r for %s"
            % (n, smiles))


def test_stereo_free_input_still_emits():
    # The change must not touch molecules with no stereo to express.
    assert _emits("C1CC11COC11CCC1") is not None  # dispiro cage, no stereocentres


def test_strip_escape_hatch_restores_old_behaviour(monkeypatch):
    # ORTHONYM_BE_STRIP_STEREO=1 restores the P-91.2.2 ship-stripped path
    # (measurement/back-compat only).
    monkeypatch.setenv("ORTHONYM_BE_STRIP_STEREO", "1")
    assert _emits("CN1[C@@H]2CC[C@H]1C[C@H](O)C2") is not None
