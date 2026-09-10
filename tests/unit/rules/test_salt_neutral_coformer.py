""" a phase Lever A2: fold a recognized neutral solvate (water of
crystallization) into an ionic-salt name instead of abstaining.

`name_salt` used to fail-closed unconditionally on ANY neutral co-fragment
(`if neutrals: return ''`). A salt with water of crystallization -- e.g.
cetylpyridinium chloride monohydrate (CHEBI:3566) -- therefore abstained
even though the ionic part names fine. This relaxes the guard to fold a
recognized water solvate as a '<salt> <mult>hydrate' suffix /
 general nomenclature) while keeping the hard fail-closed for any
UNRECOGNIZED neutral co-former (0-wrong).
"""
from orthonym import name_compound
from rdkit import Chem
from rdkit.Chem import inchi


def _rt(name):
    from orthonym.namer import _validity_gate_name_to_smiles
    smi = _validity_gate_name_to_smiles(name)
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) if smi else None


def test_ionic_salt_plus_water_names_as_hydrate():
    # CHEBI:3566 -- 1-hexadecylpyridin-1-ium chloride + water of crystallization.
    smi = "CCCCCCCCCCCCCCCC[n+]1ccccc1.O.[Cl-]"
    name = name_compound(smi, style="pin")
    assert name and "unknown" not in name
    assert _rt(name) == inchi.MolToInchiKey(Chem.MolFromSmiles(smi))


def test_pure_salt_without_neutral_unchanged():
    # regression: a plain salt with no neutral co-former still names.
    assert name_compound("[Na+].CC(=O)[O-]", style="pin") == "sodium acetate"


def test_unrecognized_neutral_coformer_still_abstains():
    # 0-wrong: an unrecognized neutral organic co-former (ethanol, not water)
    # must NOT be silently dropped or forced into a hydrate-shaped name --
    # name_salt has no mechanism to fold it, so it must fail closed.
    smi = "[Na+].CC(=O)[O-].CCO"
    assert name_salt_abstains(smi)


def name_salt_abstains(smi: str) -> bool:
    from orthonym.rules.salts import name_salt
    mol = Chem.MolFromSmiles(smi)
    return name_salt(mol, "pin") == ""


def _rt_ok_or_abstain(smi: str) -> bool:
    """True iff name_compound either abstains (unknown / '(not supported)')
    or emits a name whose OPSIN round-trip InChIKey MATCHES the input --
    i.e. it never ships a WRONG species."""
    name = name_compound(smi, style="pin")
    if not name or "unknown" in name or "not supported" in name:
        return True
    got = _rt(name)
    want = inchi.MolToInchiKey(Chem.MolFromSmiles(smi))
    return got == want


def test_aqueous_hydrohalide_fails_closed():
    # a review-found 0-wrong BLOCKER: a hydroacid written ionically ([H+].[X-])
    # plus water of crystallization used to leave the [H+] orphaned -- the
    # hydroacid-merge branch only fires when an ORGANIC neutral (>1 heavy
    # atom) is present, so a water-only neutral set never reaches it. The
    # cation guard then passed VACUOUSLY (0 == 0, since h_plus_frags
    # is excluded from cation_list), and the A2 water-fold appended
    # 'monohydrate' to an anion-only name -- shipping a WRONG species (net
    # charge -1 instead of neutral): 'O.[H+].[Cl-]' -> 'chloride monohydrate'.
    for smi in ("O.[H+].[Cl-]", "O.[H+].[Br-]", "O.O.[H+].[Cl-]"):
        assert _rt_ok_or_abstain(smi), (
            f"{smi} shipped a wrong-species name: "
            f"{name_compound(smi, style='pin')!r}"
        )


def test_ionic_hydroacid_fails_closed():
    # PRE-EXISTING sibling (not introduced by Lever A2, same root cause,
    # same fix): a bare [H+].[X-] pair with NO neutral fragment at all also
    # falls through the hydroacid-merge branch (`neutrals` is empty) and
    # used to emit an anion-only name ('chloride', 'acetate') that does not
    # round-trip to the input (which carries a net -1 charge under this
    # reading, or is really "hydrogen chloride"/"acetic acid" and not a salt
    # at all) -- either way, an anion word alone is a wrong species.
    for smi in ("[H+].[Cl-]", "[H+].CC(=O)[O-]"):
        assert _rt_ok_or_abstain(smi), (
            f"{smi} shipped a wrong-species name: "
            f"{name_compound(smi, style='pin')!r}"
        )


def test_drug_hydrochloride_still_names():
    # Regression: the LEGITIMATE Drug.[H+].[Cl-] -> 'drug hydrochloride'
    # path RETURNS inside the hydroacid-merge branch (organic neutral +
    # h_plus_frags + halide anion) and must never reach the new orphaned-H+
    # guard.
    smi = "Clc1ccccc1CCN.[H+].[Cl-]"
    name = name_compound(smi, style="pin")
    assert name and "hydrochloride" in name
    assert _rt(name) == inchi.MolToInchiKey(Chem.MolFromSmiles(smi))
