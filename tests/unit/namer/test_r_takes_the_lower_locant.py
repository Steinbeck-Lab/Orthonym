"""When the numbering is a choice, the R prefix takes the lower locant, as (j) says and after (i).

``### **** NUMBERING`` (``the Blue Book``). Criterion (g) (``:3307``), "lowest locants for the substituent
cited first as a prefix in the name", is decided by alphanumerical order, which leaves the stereodescriptors out
(``:3446``), so two prefixes that differ only in R and S TIE under (g). The choice between them is (j) (``:3346``):
"the lower locant is assigned to CIP stereodescriptors Z, R, M, and r... preferred to E, S, P, and s". Its own row is
``1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)`` (``:3399``); the next, ``(2Z,4S,8R,9E)-undeca-2,9-diene-4,8-diol
(PIN)`` (``:3403``), shows that the descriptors of the parent and those of the prefixes are compared TOGETHER, in order
of locant, at the first point of difference ("the choice is between 'E' and 'Z' for position '2', not between 'R' and
'S' for position '4'", ``:3405``). (j) comes after (h) (``:3320``) and (i) (``:3336``): "isotopic modifications have
seniority for low locants over stereodescriptors described in (j) below" (``:3346``), so the lowest locant of a label
is not given up for the R prefix.

The numbering keys compared the NAMES of the prefixes with ``alpha_sort_key``, which leaves R/S out, so the twins tied
and the order the atoms were written in put R at the higher locant (``3-[(1R)-..]-2-[(1S)-..]bicyclo[2.2.1]heptane``).
The spelling check of the best-effort tier saw it on a ring only, so the chain, bicyclo and guanidine rows were
``pin_verified`` and false. Every row reads back with OPSIN to the input structure.

Where the numbering is NOT a choice (a suffix, a heteroatom, the drawn NH of a pyrazole fixes it) R stays where the
numbering puts it, at the higher locant if need be (``:45465``): those rows are the last table. A row and its mirror
(the label on the other twin) are both in the tables, so the numbering is not right by the order of the atoms.
"""
import pytest

from orthonym import name_compound
from tests.support.jars import jar_or_skip
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

PIN = "pin_verified"
SYSTEMATIC = "systematic_verified"

#: (the numbering key that decides it, SMILES, the name, the tier of the best-effort run). The tier of a name that
#: is a PIN is ``pin_verified``; the name of a molecule whose PIN is another construction (an amine for an amino
#: prefix, a longer chain for the 2-ethylpropane) is ``systematic_verified``.
FREE_NUMBERING = [
    # rules.locants._alphabetical_tiebreaker: the orientation of a chain
    ("chain: ethane", "C[C@H](Cl)SCCS[C@@H](C)Cl",
     "1-{[(1R)-1-chloroethyl]sulfanyl}-2-{[(1S)-1-chloroethyl]sulfanyl}ethane", PIN),
    ("chain: ethane, ether", "C[C@H](Cl)OCCO[C@@H](C)Cl",
     "1-[(1R)-1-chloroethoxy]-2-[(1S)-1-chloroethoxy]ethane", PIN),
    ("chain: propane", "C[C@H](Cl)SCCCS[C@@H](C)Cl",
     "1-{[(1R)-1-chloroethyl]sulfanyl}-3-{[(1S)-1-chloroethyl]sulfanyl}propane", PIN),
    ("chain: pentane-1,5-diol", "CC[C@@H](C)C(CO)CC(CO)[C@@H](C)CC",
     "2-[(2R)-butan-2-yl]-4-[(2S)-butan-2-yl]pentane-1,5-diol", PIN),
    ("chain: butanedioate", "CC[C@@H](C)C(C(=O)OC)C(C(=O)OC)[C@@H](C)CC",
     "dimethyl 2-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]butanedioate", PIN),
    ("chain: pentane-1,3,5-triamine", "CC[C@@H](C)C(CN)C(N)C(CN)[C@@H](C)CC",
     "2-[(2R)-butan-2-yl]-4-[(2S)-butan-2-yl]pentane-1,3,5-triamine", PIN),
    ("chain: hexane-2,5-diamine", "CC[C@@H](C)C(C(C)N)C(C(C)N)[C@@H](C)CC",
     "3-[(2R)-butan-2-yl]-4-[(2S)-butan-2-yl]hexane-2,5-diamine", PIN),
    ("chain: 2-ethylpropane-1,3-diamine, N-prefixes", "CCC(CN[C@H](C)CC)CN[C@@H](C)CC",
     "N1-[(2R)-butan-2-yl]-N3-[(2S)-butan-2-yl]-2-ethylpropane-1,3-diamine", PIN),
    ("chain: 2-ethylpropane with amino prefixes", "CCC(CN[C@H](Cl)CC)CN[C@@H](Cl)CC",
     "1-{[(1R)-1-chloropropyl]amino}-3-{[(1S)-1-chloropropyl]amino}-2-ethylpropane", SYSTEMATIC),
    # decomposition.weave._number_parent
    ("weave: propan-2-ol, alkoxy prefixes", "C[C@H](Cl)OCC(O)CO[C@@H](C)Cl",
     "1-[(1R)-1-chloroethoxy]-3-[(1S)-1-chloroethoxy]propan-2-ol", PIN),
    ("weave: propan-2-ol, butan-2-yloxy", "CC[C@@H](C)OCC(O)CO[C@@H](C)CC",
     "1-{[(2R)-butan-2-yl]oxy}-3-{[(2S)-butan-2-yl]oxy}propan-2-ol", PIN),
    # rules.heterocycles._ring_prefix_names: the monocycle, the bicycle, the carbocycle of the general engine
    ("monocycle: pyridine", "C[C@H](Cl)c1cccc([C@@H](C)Cl)n1",
     "2-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl]pyridine", PIN),
    ("monocycle: thiophene", "C[C@H](Cl)c1ccc([C@@H](C)Cl)s1",
     "2-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]thiophene", PIN),
    ("monocycle: oxolane", "C[C@H](Cl)C1CCC([C@@H](C)Cl)O1",
     "2-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]oxolane", PIN),
    ("monocycle: oxirane", "C[C@H](Cl)C1OC1[C@@H](C)Cl",
     "2-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]oxirane", PIN),
    ("monocycle: piperidine", "C[C@H](Cl)C1CNCC([C@@H](C)Cl)C1",
     "3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]piperidine", PIN),
    ("monocycle: piperazine", "C[C@H](Cl)N1CCN([C@@H](C)Cl)CC1",
     "1-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl]piperazine", PIN),
    ("monocycle: 1,4-dioxane", "C[C@H](Cl)C1COC([C@@H](C)Cl)CO1",
     "2-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1,4-dioxane", PIN),
    ("monocycle: 1,3,5-triazine", "C[C@H](Cl)c1ncnc([C@@H](C)Cl)n1",
     "2-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl]-1,3,5-triazine", PIN),
    ("monocycle, two descriptors per prefix: furan", "C[C@H](Cl)[C@H](C)c1ccc([C@@H](C)[C@@H](C)Cl)o1",
     "2-[(2R,3R)-3-chlorobutan-2-yl]-5-[(2R,3S)-3-chlorobutan-2-yl]furan", PIN),
    ("bicycle: bicyclo[2.2.1]heptane", "C[C@H](Cl)C1C2CCC(C2)C1[C@@H](C)Cl",
     "2-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]bicyclo[2.2.1]heptane", PIN),
    ("bicycle: bicyclo[2.2.2]octane", "C[C@H](Cl)C1C2CCC(CC2)C1[C@@H](C)Cl",
     "2-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]bicyclo[2.2.2]octane", PIN),
    ("general engine carbocycle: amino prefixes", "C[C@H](Cl)NC1CCC(N[C@@H](C)Cl)CC1",
     "1-{[(1R)-1-chloroethyl]amino}-4-{[(1S)-1-chloroethyl]amino}cyclohexane", SYSTEMATIC),
    # rules.bridged_fused_pin._prefix_groups: a fused ring system
    ("fused: naphthalene 1,5", "C[C@H](Cl)c1cccc2c([C@@H](C)Cl)cccc12",
     "1-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]naphthalene", PIN),
    ("fused: naphthalene 2,6", "C[C@H](Cl)c1ccc2cc([C@@H](C)Cl)ccc2c1",
     "2-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl]naphthalene", PIN),
    ("fused: anthracene 9,10", "C[C@H](Cl)c1c2ccccc2c([C@@H](C)Cl)c2ccccc12",
     "9-[(1R)-1-chloroethyl]-10-[(1S)-1-chloroethyl]anthracene", PIN),
    ("fused, two descriptors per prefix: naphthalene 1,5",
     "C[C@H](Cl)[C@H](C)c1cccc2c([C@@H](C)[C@@H](C)Cl)cccc12",
     "1-[(2R,3R)-3-chlorobutan-2-yl]-5-[(2R,3S)-3-chlorobutan-2-yl]naphthalene", PIN),
    # rules.cycloalkanes._isotope_cip_key: the carbocycle of the default tier, two descriptors per prefix
    ("cyclohexane 1,4, two descriptors per prefix", "C[C@H](Cl)[C@H](C)C1CCC([C@@H](C)[C@@H](C)Cl)CC1",
     "1-[(2R,3R)-3-chlorobutan-2-yl]-4-[(2R,3S)-3-chlorobutan-2-yl]cyclohexane", PIN),
    ("cyclohexane 1,3, two descriptors per prefix", "C[C@H](Cl)[C@@H](C)C1CCCC([C@H](C)[C@@H](C)Cl)C1",
     "1-[(2S,3R)-3-chlorobutan-2-yl]-3-[(2S,3S)-3-chlorobutan-2-yl]cyclohexane", PIN),
    ("cyclohexan-1-ol 3,5, two descriptors per prefix", "C[C@H](Cl)[C@H](C)C1CC(O)CC([C@@H](C)[C@@H](C)Cl)C1",
     "3-[(2R,3R)-3-chlorobutan-2-yl]-5-[(2R,3S)-3-chlorobutan-2-yl]cyclohexan-1-ol", PIN),
    ("cyclopentan-1-one 2,5, two descriptors per prefix", "C[C@H](Cl)[C@H](C)C1CCC([C@@H](C)[C@@H](C)Cl)C1=O",
     "2-[(2R,3R)-3-chlorobutan-2-yl]-5-[(2R,3S)-3-chlorobutan-2-yl]cyclopentan-1-one", PIN),
    ("cyclohexane 1,4, oxy prefixes", "C[C@H](Cl)[C@@H](C)OC1CCC(O[C@@H](C)[C@@H](C)Cl)CC1",
     "1-{[(2R,3S)-3-chlorobutan-2-yl]oxy}-4-{[(2S,3R)-3-chlorobutan-2-yl]oxy}cyclohexane", PIN),
    ("cyclohexane 1,4, sulfanyl prefixes", "C[C@H](Cl)[C@H](C)SC1CCC(S[C@H](C)[C@@H](C)Cl)CC1",
     "1-{[(2R,3R)-3-chlorobutan-2-yl]sulfanyl}-4-{[(2S,3S)-3-chlorobutan-2-yl]sulfanyl}cyclohexane", PIN),
    # data.fused_heterocycles._select_lowest_locant_match (rules.fused_match_ties): the automorphic numberings of a core
    ("fused core: carbazole 3,6", "C[C@H](Cl)c1ccc2[nH]c3ccc([C@@H](C)Cl)cc3c2c1",
     "3-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl]-9H-carbazole", PIN),
    ("fused core: fluorene 2,7", "C[C@H](Cl)c1ccc2c(c1)Cc1cc([C@@H](C)Cl)ccc1-2",
     "2-[(1R)-1-chloroethyl]-7-[(1S)-1-chloroethyl]-9H-fluorene", PIN),
    ("fused core: dibenzo[b,d]furan 2,8", "C[C@H](Cl)c1ccc2oc3ccc([C@@H](C)Cl)cc3c2c1",
     "2-[(1R)-1-chloroethyl]-8-[(1S)-1-chloroethyl]dibenzo[b,d]furan", PIN),
    ("fused core: phenazine 2,7", "C[C@H](Cl)c1ccc2nc3cc([C@@H](C)Cl)ccc3nc2c1",
     "2-[(1R)-1-chloroethyl]-7-[(1S)-1-chloroethyl]phenazine", PIN),
    ("fused core: indane 4,7", "C[C@H](Cl)c1ccc([C@@H](C)Cl)c2CCCc12",
     "4-[(1R)-1-chloroethyl]-7-[(1S)-1-chloroethyl]-2,3-dihydro-1H-indene", PIN),
    # rules.bicyclo._bicyclo_tie_key and rules.spiro._select_spiro_numbering: (g) is 's, the twins go to (j)
    ("bicycle: Z before E (j)", "C/C=C\\C1C2CCC(C2)C1/C=C/C",
     "2-[(1Z)-prop-1-en-1-yl]-3-[(1E)-prop-1-en-1-yl]bicyclo[2.2.1]heptane", PIN),
    ("bicycle: Z before E, the other way drawn", "C/C=C/C1C2CCC(C2)C1/C=C\\C",
     "2-[(1Z)-prop-1-en-1-yl]-3-[(1E)-prop-1-en-1-yl]bicyclo[2.2.1]heptane", PIN),
    ("spiro[5.5]undecane 3,9", "C[C@H](Cl)C1CCC2(CC1)CCC([C@@H](C)Cl)CC2",
     "3-[(1R)-1-chloroethyl]-9-[(1S)-1-chloroethyl]spiro[5.5]undecane", PIN),
    ("spiro[3.5]nonane 2,7", "C[C@H](Cl)C1CCC2(CC1)CC([C@@H](C)Cl)C2",
     "2-[(1R)-1-chloroethyl]-7-[(1S)-1-chloroethyl]spiro[3.5]nonane", PIN),
    # composer._try_name_guanidine_core: the tags N and N' are a choice, R takes the N
    ("guanidine", "C[C@H](Cl)NC(=N)N[C@@H](C)Cl",
     "N-[(1R)-1-chloroethyl]-N'-[(1S)-1-chloroethyl]guanidine", PIN),
    ("guanidine, two descriptors per prefix", "C[C@H](Cl)[C@@H](C)NC(=N)N[C@H](C)[C@@H](C)Cl",
     "N-[(2R,3R)-3-chlorobutan-2-yl]-N'-[(2R,3S)-3-chlorobutan-2-yl]guanidine", PIN),
    ("guanidine, S,S against R,S", "C[C@H](Cl)[C@H](C)NC(=N)N[C@H](C)[C@H](C)Cl",
     "N-[(2R,3S)-3-chlorobutan-2-yl]-N'-[(2S,3S)-3-chlorobutan-2-yl]guanidine", PIN),
]


@pytest.mark.parametrize("site, smiles, expected, tier", FREE_NUMBERING, ids=[r[0] for r in FREE_NUMBERING])
def test_r_takes_the_lower_locant_when_the_numbering_is_a_choice(site, smiles, expected, tier):
    jar_or_skip("opsin")
    assert name_compound(smiles) == expected
    row = name_best_effort(smiles)
    assert row["name"] == expected
    assert row["tier"] == tier, row
    assert name_is_rt_exact(expected, smiles)


#: (i) (:3336) before (j) (:3346): the lowest locant of a label is not given up for the R prefix. Each label is
#: on one twin, and its mirror on the other; the R prefix is cited first and stands at the lower locant only where
#: the label leaves the choice.
ISOTOPES_BEFORE_TWINS = [
    ("chain, label on the R side", "C[C@H](Cl)SC([2H])CCS[C@@H](C)Cl",
     "3-{[(1R)-1-chloroethyl]sulfanyl}-1-{[(1S)-1-chloroethyl]sulfanyl}(1-2H1)propane"),
    ("chain, label on the S side", "C[C@H](Cl)SCCC([2H])S[C@@H](C)Cl",
     "1-{[(1R)-1-chloroethyl]sulfanyl}-3-{[(1S)-1-chloroethyl]sulfanyl}(1-2H1)propane"),
    ("chain, two labels, R side", "C[C@H](Cl)SC([2H])([2H])CCS[C@@H](C)Cl",
     "3-{[(1R)-1-chloroethyl]sulfanyl}-1-{[(1S)-1-chloroethyl]sulfanyl}(1,1-2H2)propane"),
    ("chain, two labels, S side", "C[C@H](Cl)SCCC([2H])([2H])S[C@@H](C)Cl",
     "1-{[(1R)-1-chloroethyl]sulfanyl}-3-{[(1S)-1-chloroethyl]sulfanyl}(1,1-2H2)propane"),
    ("pyridine, label by the R prefix", "C[C@H](Cl)c1c([2H])ccc([C@@H](C)Cl)n1",
     "6-[(1R)-1-chloroethyl]-2-[(1S)-1-chloroethyl](3-2H)pyridine"),
    ("pyridine, label by the S prefix", "C[C@H](Cl)c1ccc([2H])c([C@@H](C)Cl)n1",
     "2-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl](3-2H)pyridine"),
    ("thiophene 2,5", "C[C@H](Cl)c1cc([2H])c([C@@H](C)Cl)s1",
     "2-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl](3-2H)thiophene"),
    ("thiophene 2,4", "C[C@H](Cl)c1c([2H])c([C@@H](C)Cl)sc1",
     "2-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl](3-2H)thiophene"),
    ("benzene 1,3 (:3399), label by the first-drawn prefix", "c1([C@H](C)Cl)cc([C@@H](C)Cl)c([2H])cc1",
     "3-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl](4-2H)benzene"),
    ("benzene 1,3, the label on the other twin", "c1([C@@H](C)Cl)cc([C@H](C)Cl)c([2H])cc1",
     "1-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl](4-2H)benzene"),
    ("cyclohexane 1,4, label on one twin's side", "C[C@@H](Cl)C1CC([2H])C(CC1)[C@H](C)Cl",
     "4-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl](2-2H1)cyclohexane"),
    ("cyclohexane 1,4, the other twin's", "C[C@H](Cl)C1CC([2H])C(CC1)[C@@H](C)Cl",
     "1-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl](2-2H1)cyclohexane"),
    ("carbazole 3,6, label at 2", "C[C@H](Cl)c1ccc2[nH]c3cc([2H])c([C@@H](C)Cl)cc3c2c1",
     "3-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl](2-2H)-9H-carbazole"),
    ("bicyclo[2.2.1]heptane, label on the R prefix's carbon", "C[C@H](Cl)C1C2CCC(C2)C1([2H])[C@@H](C)Cl",
     "2-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl](2-2H)bicyclo[2.2.1]heptane"),
    ("bicyclo[2.2.1]heptane, label on the S prefix's carbon", "C[C@H](Cl)C1([2H])C2CCC(C2)C1[C@@H](C)Cl",
     "3-[(1R)-1-chloroethyl]-2-[(1S)-1-chloroethyl](2-2H)bicyclo[2.2.1]heptane"),
    ("spiro[5.5]undecane, label on the first-drawn prefix's carbon", "C[C@H](Cl)C1([2H])CCC2(CC1)CCC([C@@H](C)Cl)CC2",
     "9-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl](3-2H)spiro[5.5]undecane"),
    ("spiro[5.5]undecane, label on the other", "C[C@H](Cl)C1CCC2(CC1)CCC([2H])([C@@H](C)Cl)CC2",
     "3-[(1R)-1-chloroethyl]-9-[(1S)-1-chloroethyl](3-2H)spiro[5.5]undecane"),
    ("naphthalene 1,5, label at 2", "C[C@H](Cl)c1cccc2c([C@@H](C)Cl)c([2H])ccc12",
     "1-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl](2-2H)naphthalene"),
    ("naphthalene 1,5, label at 4", "C[C@H](Cl)c1cccc2c([C@@H](C)Cl)ccc([2H])c12",
     "1-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl](4-2H)naphthalene"),
    ("naphthalene 1,5, label on the other twin", "C[C@@H](Cl)c1cccc2c([C@H](C)Cl)c([2H])ccc12",
     "5-[(1R)-1-chloroethyl]-1-[(1S)-1-chloroethyl](2-2H)naphthalene"),
    ("propan-2-ol, ethers, label on one end", "C[C@H](Cl)OC([2H])C(O)CO[C@@H](C)Cl",
     "3-[(1R)-1-chloroethoxy]-1-[(1S)-1-chloroethoxy](1-2H1)propan-2-ol"),
    ("propan-2-ol, ethers, label on the other", "C[C@H](Cl)OCC(O)C([2H])O[C@@H](C)Cl",
     "1-[(1R)-1-chloroethoxy]-3-[(1S)-1-chloroethoxy](1-2H1)propan-2-ol"),
    ("propan-2-yl pentanoate, ethers (R first and lower)", "CCCCC(=O)OC(CO[C@H](C)Cl)CO[C@@H](C)Cl",
     "1-[(1R)-1-chloroethoxy]-3-[(1S)-1-chloroethoxy]propan-2-yl pentanoate"),
    ("propan-2-yl pentanoate, ethers, label on one end", "CCCCC(=O)OC(CO[C@H](C)Cl)C([2H])O[C@@H](C)Cl",
     "1-[(1R)-1-chloroethoxy]-3-[(1S)-1-chloroethoxy](1-2H1)propan-2-yl pentanoate"),
    ("propan-2-yl pentanoate, ethers, label on the other", "CCCCC(=O)OC(CO[C@@H](C)Cl)C([2H])O[C@H](C)Cl",
     "3-[(1R)-1-chloroethoxy]-1-[(1S)-1-chloroethoxy](1-2H1)propan-2-yl pentanoate"),
    ("butanediamide, label on C2, one N", "C[C@H](Cl)NC(=O)C([2H])CC(=O)N[C@@H](C)Cl",
     "N4-[(1R)-1-chloroethyl]-N1-[(1S)-1-chloroethyl](2-2H1)butanediamide"),
    ("butanediamide, label on C2, the other N", "C[C@@H](Cl)NC(=O)C([2H])CC(=O)N[C@H](C)Cl",
     "N1-[(1R)-1-chloroethyl]-N4-[(1S)-1-chloroethyl](2-2H1)butanediamide"),
    ("butanediamide, label on the N of the R prefix", "C[C@H](Cl)NC(=O)CCC(=O)N([2H])[C@@H](C)Cl",
     "N1-[(1R)-1-chloroethyl]-N4-[(1S)-1-chloroethyl](N1-2H)butanediamide"),
    ("butanediamide, label on the N of the S prefix", "C[C@@H](Cl)NC(=O)CCC(=O)N([2H])[C@H](C)Cl",
     "N4-[(1R)-1-chloroethyl]-N1-[(1S)-1-chloroethyl](N1-2H)butanediamide"),
    ("imidodicarbonimidic diamide, label on the N of the R prefix", "C[C@H](Cl)NC(=N)NC(=N)N([2H])[C@@H](C)Cl",
     "N1-[(1R)-1-chloroethyl]-N3-[(1S)-1-chloroethyl](N1-2H)imidodicarbonimidic diamide"),
    ("imidodicarbonimidic diamide, label on the N of the S prefix", "C[C@@H](Cl)NC(=N)NC(=N)N([2H])[C@H](C)Cl",
     "N3-[(1R)-1-chloroethyl]-N1-[(1S)-1-chloroethyl](N1-2H)imidodicarbonimidic diamide"),
    ("guanidine, label on the N of one prefix", "C[C@H](Cl)NC(=N)N([2H])[C@@H](C)Cl",
     "N-[(1R)-1-chloroethyl]-N'-[(1S)-1-chloroethyl](N1-2H)guanidine"),
    ("guanidine, label on the N of the other", "C[C@@H](Cl)NC(=N)N([2H])[C@H](C)Cl",
     "N'-[(1R)-1-chloroethyl]-N-[(1S)-1-chloroethyl](N1-2H)guanidine"),
    ("dimethyl butanedioate, label on one carbon", "CC[C@@H](C)C([2H])(C(=O)OC)C(C(=O)OC)[C@@H](C)CC",
     "dimethyl 2-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl](2-2H)butanedioate"),
    ("dimethyl butanedioate, label on the other", "CC[C@@H](C)C(C(=O)OC)C([2H])(C(=O)OC)[C@@H](C)CC",
     "dimethyl 3-[(2R)-butan-2-yl]-2-[(2S)-butan-2-yl](2-2H)butanedioate"),
]


@pytest.mark.parametrize("why, smiles, expected", ISOTOPES_BEFORE_TWINS, ids=[r[0] for r in ISOTOPES_BEFORE_TWINS])
def test_the_lowest_locant_of_a_label_comes_before_the_choice_of_the_twins(why, smiles, expected):
    """ (i) (:3336) "the lowest locants to the modified atoms or groups"; "isotopic modifications have seniority
    for low locants over stereodescriptors described in (j) below" (:3346); (g) leaves R and S out (:3446)."""
    jar_or_skip("opsin")
    assert name_compound(smiles) == expected
    assert name_is_rt_exact(expected, smiles)


#: (j) (:3346): the descriptors of the parent and of the prefixes are compared together, in order of locant,
#: at the first point of difference: ``(2Z,4S,8R,9E)-undeca-2,9-diene-4,8-diol (PIN)`` (:3403), "the choice is between
#: 'E' and 'Z' for position '2', not between 'R' and 'S' for position '4'" (:3405).
TWINS_WITH_PARENT_DESCRIPTORS = [
    ("heptane-2,6-diol (2R,6S): the parent's R at 2 decides", "C[C@@H](O)C([C@H](C)Cl)CC([C@@H](C)Cl)[C@H](C)O",
     "(2R,6S)-5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]heptane-2,6-diol"),
    ("heptane-2,6-diol (2R,6S), the twins the other way", "C[C@H](O)C([C@H](C)Cl)CC([C@@H](C)Cl)[C@@H](C)O",
     "(2R,6S)-3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]heptane-2,6-diol"),
    ("heptane-2,6-diol (2R,6R): the twins decide", "C[C@@H](O)C([C@H](C)Cl)CC([C@@H](C)Cl)[C@@H](C)O",
     "(2R,6R)-3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]heptane-2,6-diol"),
    ("heptane-2,6-diol (2S,6S): the twins decide", "C[C@H](O)C([C@H](C)Cl)CC([C@@H](C)Cl)[C@H](C)O",
     "(2S,6S)-3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]heptane-2,6-diol"),
    ("undeca-2,9-diene (:3403): the Z at 2 decides", "C/C=C\\C([C@H](C)Cl)CCCC([C@@H](C)Cl)/C=C/C",
     "(2Z,9E)-8-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl]undeca-2,9-diene"),
    ("undeca-2,9-diene, the twins the other way", "C/C=C/C([C@H](C)Cl)CCCC([C@@H](C)Cl)/C=C\\C",
     "(2Z,9E)-4-[(1R)-1-chloroethyl]-8-[(1S)-1-chloroethyl]undeca-2,9-diene"),
    ("undeca-2,9-diene (Z,Z): the twins decide", "C/C=C\\C([C@H](C)Cl)CCCC([C@@H](C)Cl)/C=C\\C",
     "(2Z,9Z)-4-[(1R)-1-chloroethyl]-8-[(1S)-1-chloroethyl]undeca-2,9-diene"),
    ("undeca-2,9-diene (E,E): the twins decide", "C/C=C/C([C@H](C)Cl)CCCC([C@@H](C)Cl)/C=C/C",
     "(2E,9E)-4-[(1R)-1-chloroethyl]-8-[(1S)-1-chloroethyl]undeca-2,9-diene"),
]


@pytest.mark.parametrize("why, smiles, expected", TWINS_WITH_PARENT_DESCRIPTORS,
                         ids=[r[0] for r in TWINS_WITH_PARENT_DESCRIPTORS])
def test_the_parents_descriptors_and_the_twins_are_compared_together(why, smiles, expected):
    jar_or_skip("opsin")
    row = name_best_effort(smiles)
    assert row["name"] == expected and name_compound(smiles) == expected
    assert row["tier"] == PIN, row
    assert name_is_rt_exact(expected, smiles)


#: Rows the rule does NOT move: a higher criterion fixes the numbering, or the name is already right.
UNMOVED = [
    # the descriptors are those of the parent: (j) at its own locants (:3383,:3354)
    ("parent descriptors: pentane", "C[C@H](F)C[C@@H](C)F", "(2R,4S)-2,4-difluoropentane"),
    ("parent descriptors: pentane-2,4-diol", "C[C@H](O)C[C@@H](C)O", "(2R,4S)-pentane-2,4-diol"),
    ("parent descriptors: cyclohexane", "Cl[C@H]1CCC[C@@H](Cl)C1", "(1R,3S)-1,3-dichlorocyclohexane"),
    ("parent descriptors: Z before E", "OC(=O)/C=C\\C/C=C/C(=O)O", "(2Z,5E)-hepta-2,5-dienedioic acid"),
    # one descriptor per prefix on a ring: the book's own row (:3399)
    ("benzene 1,3 (:3399)", "CC[C@@H](C)c1cccc([C@@H](C)CC)c1", "1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene"),
    ("benzene 1,4", "CC[C@@H](C)c1ccc([C@@H](C)CC)cc1", "1-[(2R)-butan-2-yl]-4-[(2S)-butan-2-yl]benzene"),
    ("cyclohexane 1,4", "CC[C@@H](C)C1CCC([C@@H](C)CC)CC1", "1-[(2R)-butan-2-yl]-4-[(2S)-butan-2-yl]cyclohexane"),
    # the suffix fixes the numbering: R is cited first at the higher locant (:45465)
    ("cyclohexan-1-one, the suffix at 1", "CC[C@@H](C)C1CCC(=O)C([C@@H](C)CC)C1",
     "4-[(2R)-butan-2-yl]-2-[(2S)-butan-2-yl]cyclohexan-1-one"),
    # the NH the structure is drawn with fixes the numbering of a 1H-pyrazole: the other numbering is the other
    # tautomer, which the standard InChIKey does not tell apart (``### **** NUMBERING`` criterion (b),
    # indicated hydrogen, ``:3246``, comes before (g))
    ("1H-pyrazole, butan-2-yl", "CC[C@@H](C)c1cc([C@@H](C)CC)n[nH]1",
     "5-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]-1H-pyrazole"),
    ("1H-pyrazole, 1-chloroethyl", "C[C@H](Cl)c1cc([C@@H](C)Cl)[nH]n1",
     "5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]-1H-pyrazole"),
    # (g) before (j): the two prefixes differ in the letters, the one cited first takes the lower locant
    ("different prefixes keep the order of the letters", "C[C@H](Cl)C1CCC(CC1)C[C@@H](C)Cl",
     "1-[(1S)-1-chloroethyl]-4-[(2R)-2-chloropropyl]cyclohexane"),
]


@pytest.mark.parametrize("site, smiles, expected", UNMOVED, ids=[r[0] for r in UNMOVED])
def test_the_numbering_that_is_not_a_choice_is_not_moved(site, smiles, expected):
    jar_or_skip("opsin")
    assert name_compound(smiles) == expected
    assert name_is_rt_exact(expected, smiles)
