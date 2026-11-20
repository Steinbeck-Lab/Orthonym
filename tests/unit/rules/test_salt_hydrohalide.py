"""A salt drawn beside a neutral hydrogen halide is named with the word 'hydrochloride' (C4 mixtures, 2026-11-20).

 (the Blue Book ff.), method (3): "for salts of hydrohalogen acids only, the unaltered name of the base is
followed by hydrofluoride, hydrobromide, hydrochloride, or hydroiodide" ('lysine monohydrochloride',:54550). A
traditional method, so the name is labelled below the PIN, as the 'hydrate' of the same step (``rules.salts``). ChEBI
draws thiamine chloride hydrochloride as the cation, a chloride and a neutral HCl; the salt namer refused every neutral
part but water, and the drawing got no name."""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("C[N+](C)(C)C.[Cl-].Cl", "N,N,N-trimethylmethanaminium chloride monohydrochloride"),
    ("C[N+](C)(C)C.[Cl-].Cl.O", "N,N,N-trimethylmethanaminium chloride monohydrochloride monohydrate"),
    ("Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1.Cl.[Cl-]",
     "3-[(4-amino-2-methylpyrimidin-5-yl)methyl]-5-(2-hydroxyethyl)-4-methyl-1,3-thiazol-3-ium chloride "
     "monohydrochloride"),
])
def test_a_salt_beside_a_hydrogen_halide_takes_the_hydrohalide_word(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "systematic_verified", False), row
    assert tier_row(smiles, "pin")["tier"] == "abstain"
    assert_full_rt(name, smiles)


def test_any_other_neutral_part_still_fails_closed_in_the_salt_namer():
    from rdkit import Chem

    from orthonym.rules.salts import name_salt
    assert name_salt(Chem.MolFromSmiles("C[N+](C)(C)C.[Cl-].CO")) in ("", None)


@pytest.mark.parametrize("smiles", ["CC(=O)[O-].[Na+].Cl", "[Na+].[Cl-].Cl", "[Cl-].[NH4+].Cl", "C[NH3+].CC(=O)[O-].Cl"])
def test_the_word_is_for_salts_of_organic_bases_with_hydrohalogen_acids_only(smiles):
    # review 2026-11-20: covers salts of organic bases, method (3) hydrohalogen acid salts only
    row = tier_row(smiles, "best-effort")
    assert "hydrochloride" not in (row["name"] or ""), row
