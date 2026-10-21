"""The choice among the charge forms of a cation applies the Blue Book order, at the best-effort tier.

Resonance fix RH gave the best-effort tier the other drawings of a delocalised cation, and the choice among
the forms' names took the form that comes first by canonical SMILES wherever the names differ in more than
their locant digits: 41 names changed against main, 12 of them for the worse (audit
`internal notes`, "Rebased end checks", Step 4). ``assembly.form_choice``
now ranks the names (``tests/unit/assembly/test_form_choice.py`` pins the rules on strings):

* CHOICE OF A PARENT STRUCTURE (the Blue Book), "(a) parent with the maximum number of cationic,
  including cationic suffixes" (:42334): rows 4, 17, 21.
* NUMBERING (:3219) (c) "principal characteristic groups and free valences (suffixes)": rows 2, 29;
   (:16633) "(1) a minimum number of compound locants" (:16635), "(2) when comparing double bond
  locants that also include compound locants, any number in parentheses is ignored" (:16655): rows 1, 12, 26.
* the names whose skeletons differ, by the fewer constructs the engine records as not PIN
  "Five-membered ring requirement" (:23708), "When fusion names are not allowed, unsaturated von Baeyer ring
  system names are preferred IUPAC names" (:23710);,:23682; "General rule for
  systematically naming cationic centers in parent hydrides",:41366): rows 3, 34, 35.
* ``_FORMS_MAX_WORK`` counts the places of the +1, not the forms (the Kekule structures of one placement, which
  RDKit's enumeration did not give, count once): rows 13, 12, 22.

The row numbers are those of ``REBASE-best-effort-renames.tsv``. Every name is read back by OPSIN's own
StdInChIKey against RDKit's key of the input.
"""
import re

import pytest
from rdkit import Chem

from orthonym import Orthonym, namer
from orthonym.cli import _emit_tier_flags
from orthonym.perception.charge_shift import charge_places, stereo_elements
from tests.unit.namer.leads_l1_support import opsin_key, rdkit_key

pytestmark = pytest.mark.opsin_gate

TIER = "best-effort"

#: (row, SMILES, the name, the criterion that decides): the 12 renames that were worse than main's, and now
#: have the name main gave before RH (rows 12 and 22 below are other names)
WORSE = [
    ("1", "C1=CC=C(C=C1)C2=C3C=CC(=C(C4=C(C5=C(C6=CC=C(N6C5=N4)C(=C7C=CC2=N7)C8=CC=CC=C8)C9=CC=CC=C9)[N+]1=CC=CC=C1)C1=CC=CC=C1)N3",
     "1-[4,9,14,19-tetraphenyl-2,21,23,24-tetraazahexacyclo[13.5.1.1^3,20.1^5,8.1^10,13.0^18,21]tetracosa-1,3(22),4,6,8,10(23),11,13,15,17,19-undecaen-22-yl]pyridin-1-ium",
     "P-31.1.4.2 (2): the ene set 1,3,4,6,8,10,11,13,15,17,19 is lower than 1,3,5,7,9,11,13,14,16,18,20 (2 compound locants each)"),
    ("2", "C1=CC=C(C=C1)C2=CC(=CC=C2)N3C4=C(C=C(C=C4)C5=CC=CC=C5N6C7=C(C=C(C=C7)N8C9=CC=CC=C9C1=C8N=CC=C1)C1=CC=[N+]=C16)C1=C3NC=C1",
     "8-(7-{2-[7-(3-phenylphenyl)-5,7-diazatricyclo[6.4.0.0^2,6]dodeca-1(12),2(6),3,8,10-pentaen-11-yl]phenyl}-5,7-diazatricyclo[6.4.0.0^2,6]dodeca-1(8),2,4,5,9,11-hexaen-5-ium-11-yl)-6,8-diazatricyclo[7.4.0.0^2,7]trideca-1(13),2,4,6,9,11-hexaene",
     "P-14.4 (c): the ium of the von Baeyer ring is at 5 against 7"),
    ("3", "C1=CC=C(C=C1)C2=NC(=[NH+]C(=N2)C3=C=C=CC=C3)C4=CC=CC(=C4)C5=C(C=CC6=CC=CC=C65)OS(=[OH+])(=O)C(C(C(C(F)(F)F)(F)F)(F)F)(F)F",
     "1-{3-[6-(cyclohexa-1,2,3,5-tetraen-1-yl)-4-phenyl-1,3,5-triazin-1-ium-2-yl]phenyl}-2-[(2,2,3,3,4,4,5,5,5-nonafluoro-1-oxidaniumylidene-1-oxo-1-thiapentan-1-yl)oxy]naphthalene",
     "fewer non-PIN parts: naphthalene and 1,3,5-triazin-1-ium, not bicyclo[4.4.0]deca-pentaene and 2,4,6-triazacyclohexatriene (P-52.2.4.1, P-52.2.2.2)"),
    ("4", "C1=CC=C2C(=C1)C=CC=C2C3=[N+](C(=N3)N4C5=CC=CC=C5C6=C4C7=CC=CC=C7C8=C6C9=CC=CC=C9C=C8)C1=CC=CC2=CC=CC=C21",
     "9-[1,4-di(naphthalen-1-yl)-1,2-dihydro-1,3-diazet-2-ylidene]-9-azahexacyclo[15.8.0.0^2,10.0^3,8.0^11,16.0^20,25]pentacosa-1(25),2(10),3,5,7,11,13,15,17,19,21,23-dodecaen-9-ium",
     "P-73.7 (a): the parent holds the cation (azahexacyclo...-9-ium)"),
    ("13", "CC1=C(C2=C(N1C3=CC=CC(=C3)C(F)(F)F)C=C=C=C2)C4=C(/C(=C\\5/C(=[N+](C6=CC=CC=C65)C7=CC=CC(=C7)C(F)(F)F)C)/C4=O)OC",
     "9-(2-methoxy-3-{2-methyl-1-[3-(trifluoromethyl)phenyl]-1H-indol-3-yl}-4-oxocyclobut-2-en-1-ylidene)-8-methyl-7-[3-(trifluoromethyl)phenyl]-7-azabicyclo[4.3.0]nona-1,5,7-trien-3-yn-7-ium",
     "the form set is no longer over the work bound (2 places of the +1, 4 forms); the name has no (1,1,1-trifluoromethyl)"),
    ("17", "CCCCCCCCCCCC[N+]1=C(C(C2=C1C=CC3=CC=CC=C32)(C)C)/C=C/4\\C(=C(C4=O)/C=C/5\\C(C6=C(N5CC)C=CC(=C6)/C=C(\\C#N)/C(=O)OC(C)(C)C)(C)C)[O-]",
     "4-{[(1E)-3-({(2E)-5-[(1E)-3-tert-butoxy-2-cyano-3-oxoprop-1-en-1-yl]-1-ethyl-3,3-dimethyl-2,3-dihydro-1H-indol-2-ylidene}methyl)-2-oxido-4-oxocyclobut-2-en-1-ylidene]methyl}-5-dodecyl-3,3-dimethyl-5-azatricyclo[7.4.0.0^2,6]trideca-1(13),2(6),4,7,9,11-hexaen-5-ium",
     "P-73.7 (a): the parent holds the cation (azatricyclo...-5-ium)"),
    ("21", "CC\\1(C2=C(C=CC(=C2)S(=O)(=O)O)N(/C1=C/C3=C(C(=CC4=[N+](C5=C(C=C4)C=C(C=C5)[N+](=O)[O-])C)C3=O)O)CCCCS(=O)(=O)O)C",
     "3-[(3-{[(2E)-3,3-dimethyl-5-sulfo-1-(4-sulfobutyl)-2,3-dihydro-1H-indol-2-ylidene]methyl}-2-hydroxy-4-oxocyclobut-2-en-1-ylidene)methyl]-2-methyl-8-nitro-2-azabicyclo[4.4.0]deca-1(6),2,4,7,9-pentaen-2-ium",
     "P-73.7 (a): the parent holds the cation (2-azabicyclo[4.4.0]...-2-ium)"),
    ("26", "CN1C=CC=CC1=C2C3=NC(=C4C=CC(=C(C5=CC=C(N5)C(=C6C=CC2=N6)C7=C(C(=C(C(=C7F)F)F)F)F)C8=CC=CC=[N+]8C)[NH2+]4)C=C3",
     "6-(1-methyl-1,2-dihydropyridin-2-ylidene)-16-(1-methylpyridin-1-ium-2-yl)-11-(pentafluorophenyl)-20,21,22,23-tetraazapentacyclo[15.2.1.1^2,5.1^7,10.1^12,15]tricosa-1,3,5(23),7(22),8,10,12,14,16,18-decaen-20-ium",
     "P-31.1.4.2 (1): 2 compound locants in the ene list of the 20-ium macrocycle against 3"),
    ("34", "C[N+]1=CC=C(C=C1)C2=CC=C(C=C2)NC(=O)C3=CC=C(C=C3)N=NNC4=CC=C(C=C4)C5=CC=[N+](C=C5)C",
     "1-({[4-(1-methylpyridin-1-ium-4-yl)phenyl]amino}diazenyl)-4-({[4-(1-methylpyridin-1-ium-4-yl)phenyl]amino}(oxo)methyl)benzene",
     "fewer non-PIN parts: 1-methylpyridin-1-ium-4-yl, not 1-azamethan-1-ium-1-yl and 1,4-dihydropyridin-4-ylidene (P-73.1.1.2)"),
    ("35", "C[N+]1=CC=C(C=C1)C=CC2=CC=C(C=C2)NC3=NC(=NC(=N3)N(CCO)CCO)NC4=CC=C(C=C4)C=CC5=CC=[N+](C=C5)C.[Cl-].[Cl-]",
     "4-{[4-[bis(2-hydroxyethyl)amino]-6-({4-[2-(1-methylpyridin-1-ium-4-yl)ethen-1-yl]phenyl}amino)-1,3,5-triazin-2-yl]amino}-1-[2-(1-methylpyridin-1-ium-4-yl)ethen-1-yl]benzene dichloride",
     "fewer non-PIN parts, as row 34"),
]

#: (row of the 333 multi-form rows of the union, SMILES, the name, what the input configures): the names of the two
#: drawings of a polymethine cation, one that keeps the E/Z of the input's double bonds and one whose Kekule
#: structure has moved them. STERODESCRIPTORS (the Blue Book), "The configuration of an organic compound
#: is systematically indicated by one or more affixes added to a name that does not itself prescribe configuration"
#: (:44570); (:44635), "From nine-membered rings onwards, stereodescriptors are required" (:44637). The
#: moved form has the parent that holds the cation (a),:42334) and names no E/Z at all, and the standard
#: InChIKey does not see these bonds (a polymethine has no /b layer), so only the count of the stereo elements a
#: form keeps (``charge_shift.kept_stereo``, ranked first by ``form_choice.beats``) separates the two names.
STEREO = [
    ("r1863", "CC1(C2=C(N=C1/C=C/C=C(/C=C/C=C\\3/C(C4=C(N3CCCCS(=O)(=O)[O-])C=CC(=C4)S(=O)(=O)[O-])(C)C)\\Cl)[N+](=CC(=C2)Cl)CCCCS(=O)(=O)[O-])C.[Na+].[Na+]",
     "disodium (2Z)-2-{(2E,4Z,6E)-4-chloro-7-[4-chloro-7,7-dimethyl-2-(4-sulfonatobutyl)-2,9-diazabicyclo[4.3.0]nona-1(6),2,4,8-tetraen-2-ium-8-yl]hepta-2,4,6-trien-1-ylidene}-3,3-dimethyl-5-sulfonato-1-(4-sulfonatobutyl)-2,3-dihydro-1H-indole",
     "4 E/Z bonds of the heptatriene bridge between two indolenine-type ends"),
    ("r2090", "CC1(C2=C(N=C1/C=C/C=C/C=C/3\\C(C4=C(N3CCCS(=O)(=O)O)C=CC(=C4)S(=O)(=O)O)(C)C)[N+](=CC(=C2)C(=O)NOCCCCCC(=O)ON5C(=O)CCC5=O)CCCS(=O)(=O)C)C",
     "(2E)-2-[(2E,4E)-5-{4-[({6-[(2,5-dioxopyrrolidin-1-yl)oxy]-6-oxohexyl}oxy)carbamoyl]-2-(4,4-dioxo-4-thiapentan-1-yl)-7,7-dimethyl-2,9-diazabicyclo[4.3.0]nona-1(6),2,4,8-tetraen-2-ium-8-yl}penta-2,4-dien-1-ylidene]-3,3-dimethyl-5-sulfo-1-(3-sulfopropyl)-2,3-dihydro-1H-indole",
     "3 E/Z bonds of the pentadiene bridge"),
]

#: the two rows that name a form main never named: the form set of an over-bound row is now named
CHANGED = [
    ("12", "CC1=C(C2=C(C=C1)C3=CC=CC=C3C4=CC=CC=C4C5=CC=CC=C52)C6=[N+](C=CN6C7=C(C=C(C=C7C8=CC=CC=C8)C9=CC=CC=C9)C1=CC=CC=C1)C",
     "1-methyl-2-(4-methylpentacyclo[18.4.0.0^2,7.0^8,13.0^14,19]tetracosa-1,3,5,7,9,11,13,15,17,19,21,23-dodecaen-3-yl)-3-(2,4,6-triphenylphenyl)-1H-imidazol-3-ium",
     "P-31.1.4.2 (1): no compound locant in the ene list (0 against 1); the 1H-imidazol-3-ium spelling and the ium at 3 are as before"),
    ("22", "CN(C)C1N2C3=CC=C2C(=C4[N+]1=C(C=C4)C(=C5C=CC(=C(C6=NC(=C3C7=CC=CC=C7)C=C6)C8=CC=CC=C8)N5)C9=CC=CC=C9)C1=CC=CC=C1",
     "21-(dimethylamino)-5,10,15,23-tetraphenyl-20,22,24,25-tetraazahexacyclo[17.3.1.1^6,9.1^11,14.0^4,22.0^16,20]pentacosa-1(23),2,4,6(25),7,9,11,13,15,17,19-undecaen-20-ium",
     "P-14.4 (c): the ium of the macrocycle at 20 against 22 (6 forms, 3 places of the +1: 52 x 3 is within the bound)"),
]

#: rows that RH renamed for the better, and that stay: (row, SMILES, the name, the tier)
KEPT = [
    ("0", "C1=CC(=CC=C1C#N)N2C=C[N+](=C2)C3=NC(=NC(=O)N3)[N+]4=CN(C=C4)C5=CC=C(C=C5)C#N",
     "1-cyano-4-(3-{4-[3-(4-cyanophenyl)-1H-imidazol-3-ium-1-yl]-6-oxo-1,6-dihydro-1,3,5-triazin-2-yl}-1H-imidazol-3-ium-1-yl)benzene",
     "systematic_verified"),
    ("14", "CC1=CC(=[N+](C=C1C2=CC=CC=C2C3=CC(=CC(=C3)C4=CC=CC=C4C5=CC=C(C=C5)N6C=[N+](C7=NC=CN=C76)C)C8=CC=CC=C8C9=C[N+](=C(C=C9C)C1=CC=CC=C1)C)C)C1=CC=CC=C1",
     "7-[4-(2-{3,5-bis[2-(1,4-dimethyl-6-phenylpyridin-1-ium-3-yl)phenyl]phenyl}phenyl)phenyl]-9-methyl-2,5,7,9-tetraazabicyclo[4.3.0]nona-1,3,5,7-tetraen-7-ium",
     "systematic_verified"),
    ("20", "CCN(CC)CCNC(=O)C1=CC(=CC=C1)S(=NC(=O)C2=CN=CC(=C2)C#CC3=CC(=CC=C3)NC(=O)C4=C[N+](=CN4C)C)(=O)C",
     "4-{[3-(2-{5-[3-(3-{[2-(diethylamino)ethyl]carbamoyl}phenyl)-3-methyl-1-oxo-4-oxa-3λ6-thia-2-azabuta-2,3-dien-1-yl]pyridin-3-yl}eth-1-yn-1-yl)phenyl]carbamoyl}-1,3-dimethyl-1H-imidazol-3-ium",
     "systematic_verified"),
    ("24", "CN1C=CC(=C2C3=NC(=C(C4=NC(=C(C5=CC=C([N-]5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=[N+](C=C8)C)C=C4)C9=CC=CC=C9)C=C3)C=C1",
     "17-(1-methyl-1,4-dihydropyridin-4-ylidene)-7-(1-methylpyridin-1-ium-4-yl)-2,12-diphenyl-21,22,23,24-tetraazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]tetracosa-1(20),2,4,6(24),7,9,11(23),12,14,16(22),18-undecaen-21-ide",
     "systematic_verified"),
    ("27", "CN1c2cc3c(cc2C(CS(=O)(=O)O)=CC1(C)C)C(c1cc(C(=O)ON2C(=O)CCC2=O)ccc1C(=O)O)=c1cc2c(cc1O3)=[N+](C)C(C)(C)C=C2CS(=O)(=O)[O-]",
     "13-(2-carboxy-5-{[(2,5-dioxopyrrolidin-1-yl)oxy](oxo)methyl}phenyl)-6,7,7,19,19,20-hexamethyl-9-(sulfomethyl)-17-(sulfonatomethyl)-2-oxa-6,20-diazapentacyclo[12.8.0.0^3,12.0^5,10.0^16,21]docosa-1(22),3,5,8,10,12,14,16(21),17-nonaen-6-ium",
     "systematic_verified"),
    # row 30: the hydrazinyl unit holds one ylidene, which can stand on the far nitrogen only, so the
    # prefix cites no locant: "Omission of locants" (the Blue Book)
    # 'ethylidenehydrazinyl (preferred prefix) (not 2-ethylidenehydrazin-1-yl)' (:2959); the name read
    # '(2-{...}hydrazinyl)' before, a locant of the unit without the free valence's
    ("30", "C[N+]1=C2C=CC=CN2C=C1C3=CC=C(C=C3)/C=N/NC(=O)N/N=C\\C4=CC=C(C=C4)C5=CN6C=CC=CC6=[N+]5C",
     "7-methyl-8-{4-[(1Z)-{[({(1E)-[4-(7-methyl-1,7-diazabicyclo[4.3.0]nona-1,3,5,8-tetraen-1-ium-8-yl)phenyl]methylidene}hydrazinyl)(oxo)methyl]hydrazinylidene}methyl]phenyl}-1,7-diazabicyclo[4.3.0]nona-1,3,5,8-tetraen-1-ium",
     "systematic_verified"),
    ("33", "C[N+]1=CC=C(C=C1)C2=C3C=CC(=C(C4=NC(=C(C5=CC=C(O5)C(=C6C=CC2=N6)C7=CC=CC=C7)C8=CC=CC=C8)C=C4)C9=CC=[N+](C=C9)C)N3",
     "12-(1-methyl-1,4-dihydropyridin-4-ylidene)-7-(1-methylpyridin-1-ium-4-yl)-2,17-diphenyl-21-oxa-22,23,24-triazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]tetracosa-1(20),2,4,6(24),7,9,11(23),13(22),14,16,18-undecaen-23-ium",
     "systematic_verified"),
    ("t0", "CN\\1C=CS/C1=C\\2/C3=NC(=CC4=NC(=C(C5=NC(=CC6=CC=C2N6)C=C5)C7=[N+](C=CS7)C)C=C4)C=C3",
     "3-methyl-2-[12-(3-methyl-2,3-dihydro-1,3-thiazol-2-ylidene)-21,22,23,24-tetraazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]tetracosa-1(20),2,4,6(24),7,9,11(23),13(22),14,16,18-undecaen-2-yl]-1,3-thiazol-3-ium",
     "pin_unverified"),
    ("t1", "C[n+]1c(-c4ccc(cc4)-c4ccc(cc4)-c3c(N=[N+]=[N-])c(N=[N+]=[N-])c(N=[N+]=[N-])c(N=[N+]=[N-])c3)[nH]c2ccccc21",
     "1-methyl-2-[4-(2',3',4',5'-tetraazido[1,1'-biphenyl]-4-yl)phenyl]-1H-1,3-benzimidazol-3-ium",
     "pin_verified"),
]

SLOW_ROW_29 = ("C[C@@H]1CN2C3=CC(=C(C4=C3N(CCCOCC5=C(C=[N+](N5)C)C6=C(C=CC7=C6N1C(=C7CCCOC8CC(=CC(=C8)C(F)F)C(F)F)C2=O)Cl)C(=C4)C(=NC=NC)N)OC)OC", "(4R)-24-(amino{[(methylimino)methyl]imino}methyl)-7-(3-{[3,5-bis(difluoromethyl)cyclohexa-2,4-dien-1-yl]oxy}propyl)-11-chloro-27,28-dimethoxy-4,15-dimethyl-32-oxo-19-oxa-2,5,15,16,23-pentaazaheptacyclo[21.6.1.1^2,6.1^5,8.0^13,17.0^26,30.0^12,31]dotriaconta-1(30),6,8(31),9,11,13(17),14,24,26,28-decaen-15-ium")
ROW_12 = "CC1=C(C2=C(C=C1)C3=CC=CC=C3C4=CC=CC=C4C5=CC=CC=C52)C6=[N+](C=CN6C7=C(C=C(C=C7C8=CC=CC=C8)C9=CC=CC=C9)C1=CC=CC=C1)C"
ROW_13 = "CC1=C(C2=C(N1C3=CC=CC(=C3)C(F)(F)F)C=C=C=C2)C4=C(/C(=C\\5/C(=[N+](C6=CC=CC=C65)C7=CC=CC(=C7)C(F)(F)F)C)/C4=O)OC"
ROW_4 = "C1=CC=C2C(=C1)C=CC=C2C3=[N+](C(=N3)N4C5=CC=CC=C5C6=C4C7=CC=CC=C7C8=C6C9=CC=CC=C9C=C8)C1=CC=CC2=CC=CC=C21"
ROW_26 = "CN1C=CC=CC1=C2C3=NC(=C4C=CC(=C(C5=CC=C(N5)C(=C6C=CC2=N6)C7=C(C(=C(C(=C7F)F)F)F)F)C8=CC=CC=[N+]8C)[NH2+]4)C=C3"


def _engine():
    return Orthonym(style="pin", **_emit_tier_flags(TIER))


def _forms_named(monkeypatch):
    """The number of forms each run of the choice names, one entry per run."""
    counts = []
    original = Orthonym._name_with_form_choice

    def spy(self, fn, forms, args, kwargs):
        counts.append(len(forms))
        return original(self, fn, forms, args, kwargs)

    monkeypatch.setattr(Orthonym, "_name_with_form_choice", spy)
    return counts


@pytest.mark.parametrize("row, smiles, name, why", WORSE + CHANGED)
def test_the_choice_gives_the_name_the_blue_book_order_ranks_first(monkeypatch, row, smiles, name, why):
    counts = _forms_named(monkeypatch)
    result = _engine().name_tiered(smiles)
    assert (result["name"], result["tier"]) == (name, "systematic_verified"), (row, why)
    assert counts and counts[0] >= 2, "the choice must run over two forms or more: %r" % counts
    assert opsin_key(name) == rdkit_key(smiles), name


@pytest.mark.parametrize("row, smiles, name, why", STEREO)
def test_the_form_that_keeps_the_configuration_of_the_input_is_named(monkeypatch, row, smiles, name, why):
    counts = _forms_named(monkeypatch)
    result = _engine().name_tiered(smiles)
    assert (result["name"], result["tier"]) == (name, "systematic_verified"), (row, why)
    assert counts and counts[0] == 2, counts
    # every double bond the input configures is a descriptor of the name
    configured = sum(1 for kind, _atoms in stereo_elements(Chem.MolFromSmiles(smiles)) if kind == "=")
    assert configured >= 3 and len(re.findall(r"(?<=[(,])\d+[EZ](?=[,)])", name)) == configured, (row, name)
    assert opsin_key(name) == rdkit_key(smiles), name


@pytest.mark.parametrize("row, smiles, name, tier", KEPT)
def test_the_renames_that_were_better_stay(row, smiles, name, tier):
    result = _engine().name_tiered(smiles)
    assert (result["name"], result["tier"]) == (name, tier), row
    assert opsin_key(name) == rdkit_key(smiles), name


@pytest.mark.slow
def test_a_macrocycle_with_the_ium_at_15_keeps_it_there():
    # row 29, 25 s: (c), 'ium' at 15 against 16
    smiles, name = SLOW_ROW_29
    result = _engine().name_tiered(smiles)
    assert (result["name"], result["tier"]) == (name, "systematic_verified")
    assert opsin_key(name) == rdkit_key(smiles)


@pytest.mark.parametrize("smiles", [ROW_26, pytest.param(ROW_4, marks=pytest.mark.slow)])
def test_every_drawing_of_the_species_gets_the_one_name(smiles):
    # the contributor guide core principle 4: the name of a species does not depend on where its SMILES draws the charge
    # (the two inputs have no stereo centre or double bond whose configuration a form string would drop)
    forms = namer._charge_forms(smiles)
    assert forms is not None and len(forms) >= 2
    names = {_engine().name_tiered(string)["name"] for string, _order in forms}
    assert len(names) == 1, names


def test_the_work_of_the_choice_counts_the_places_of_the_charge():
    # row 13: 47 heavy atoms, 4 forms (2 places of the +1, each with 2 Kekule structures of the part that
    # RDKit does not aromatise): 47 x 4 is over the bound of 160 and 47 x 2 is not
    forms = namer._charge_forms(ROW_13)
    assert forms is not None and len(forms) == 4
    heavy = Chem.MolFromSmiles(ROW_13).GetNumHeavyAtoms()
    assert charge_places(forms) == 2
    assert heavy * len(forms) > namer._FORMS_MAX_WORK >= heavy * charge_places(forms)


def test_the_places_of_the_charge_of_forms_that_differ_in_bond_orders_only_are_one():
    # row 12: 55 heavy atoms, 4 forms, 2 places
    forms = namer._charge_forms(ROW_12)
    assert forms is not None and len(forms) == 4 and charge_places(forms) == 2
    # a guanidinium has one place per nitrogen that can hold the +1, and the forms are as many as the places
    guanidinium = namer._charge_forms("CNC(=[NH2+])NCc1ccccc1")
    assert guanidinium is not None and charge_places(guanidinium) == len(guanidinium) == 3
