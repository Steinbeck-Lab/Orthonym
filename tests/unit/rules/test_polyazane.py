"""Unit tests for the polyazane parent-hydride namer (a phase).

 / /: chains of N atoms joined by N-N bonds.
hydrazine / diazene are retained PINs; longer members systematic. Azo R-N=N-R is
a substituted diazene. Fail-closed on amines / diamines / hydroxylamine /
hydrazones / azides / rings.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.polyazane import _parent_scope_has_numeral, name_polyazane


def _name(smiles):
    return name_polyazane(Chem.MolFromSmiles(smiles))


class TestBareSaturated:
    @pytest.mark.parametrize("smiles,expected", [
        ("NN", "hydrazine"),
        ("NNN", "triazane"),
        ("NNNN", "tetraazane"),
        ("NNNNN", "pentaazane"),
        ("NNNNNN", "hexaazane"),
    ])
    def test_saturated(self, smiles, expected):
        assert _name(smiles) == expected


class TestBareUnsaturated:
    @pytest.mark.parametrize("smiles,expected", [
        ("N=N", "diazene"),
        # (d) (the Blue Book; verbatim example `H2N-N=NH triazene`,:2937):
        # an unsubstituted monounsaturated homogeneous di-/trinuclear chain omits
        # the double-bond locant, so `triazene`, NOT `triaz-1-ene`.
        ("N=NN", "triazene"),
        # tetraazene keeps its locant: tetraaz-1-ene and tetraaz-2-ene are
        # different compounds (n=4 > trinuclear), so the locant is essential.
        ("N=NNN", "tetraaz-1-ene"),
    ])
    def test_unsaturated(self, smiles, expected):
        assert _name(smiles) == expected


class TestSubstituted:
    @pytest.mark.parametrize("smiles,expected", [
        ("CNN", "methylhydrazine"),              # mono -> no locant (BB phenylhydrazine precedent)
        ("CCNN", "ethylhydrazine"),
        ("c1ccccc1NN", "phenylhydrazine"),
        # hydrazine (N-N) KEEPS its locants: each N takes 2 subs, so 1,1- vs 1,2-
        # is a real distinction;.
        ("CNNC", "1,2-dimethylhydrazine"),
        ("CN(C)N", "1,1-dimethylhydrazine"),
        ("CN=N", "methyldiazene"),               # mono diazene -> no locant
        # diazene (N=N) OMITS its locants: each N is =N- with ONE substitutable
        # valence, so two substituents are necessarily 1,2 -> unambiguous ->
        # omitted. symmetric = di{R}diazene
        # (BB `dimethyldiazene (PIN)`, `diphenyldiazene (PIN)`).
        ("CN=NC", "dimethyldiazene"),
        ("CCN=NCC", "diethyldiazene"),
        ("c1ccccc1N=Nc1ccccc1", "diphenyldiazene"),  # BB PIN (azobenzene is the non-PIN)
        # unsymmetric = alphabetical parenthesised prefixes, no
        # locants (BB `ethenyl(methyl)diazene (PIN)`).
        ("C=CN=NC", "ethenyl(methyl)diazene"),
    ])
    def test_substituted(self, smiles, expected):
        assert _name(smiles) == expected


class TestDiazeneStereo:
    """ "NAMING OF STEREOISOMERS" (the Blue Book heading; decisive
    sentence:44643): the lone skeletal N=N E/Z descriptor is cited with a locant
    "when such locants are present". The 2-N diazene constitution cites no
    numeral omits the substituent locants; (d) elides the
    parent N=N locant), so the descriptor locant is OMITTED -> `(Z)-` /`(E)-`.
    This is the bb_conformance target `(Z)-diphenyldiazene`, and it
    mirrors `(E)-cyclooctene` (_handler_shared.py:1532)."""

    @pytest.mark.parametrize("smiles,expected", [
        # THE bb_conformance target: symmetric diaryl diazene, no
        # parent numeral -> descriptor locant omitted.
        ("c1ccc(/N=N\\c2ccccc2)cc1", "(Z)-diphenyldiazene"),
        # symmetric dialkyl, both geometries.
        ("C/N=N/C", "(E)-dimethyldiazene"),
        ("C/N=N\\C", "(Z)-dimethyldiazene"),
        # unsymmetric: alphabetical parenthesised prefixes, still no parent
        # numeral -> descriptor locant omitted.
        ("C/N=N/c1ccccc1", "(E)-methyl(phenyl)diazene"),
    ])
    def test_ez_locant_omitted(self, smiles, expected):
        assert _name(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # No E/Z assigned in the input -> no descriptor injected (the
        # constitution is unchanged; the injection is purely additive).
        ("c1ccccc1N=Nc1ccccc1", "diphenyldiazene"),
        ("CN=NC", "dimethyldiazene"),
        # Saturated N-N (hydrazine): no chain double bond -> never a descriptor,
        # and its cited 1,2- locants are untouched.
        ("CNNC", "1,2-dimethylhydrazine"),
    ])
    def test_no_descriptor_when_no_ez(self, smiles, expected):
        assert _name(smiles) == expected


class TestParentScopeNumeral:
    """ "Citation of locants" (the Blue Book) scopes locants per
    enclosing-mark unit, so the name-dependent test counts a numeral only
    at PARENT scope. A digit inside `(...)`/`[...]` (e.g. the `2` of
    `naphthalen-2-yl`) belongs to the substituent, not the parent."""

    @pytest.mark.parametrize("name,has_numeral", [
        # OMIT side: no parent numeral -> bare `(Z)-`/`(E)-`.
        ("diphenyldiazene", False),
        ("dimethyldiazene", False),
        ("methyl(phenyl)diazene", False),
        # scoping: the enclosed `2` is the substituent's, not the parent's.
        ("(naphthalen-2-yl)(phenyl)diazene", False),
        # KEEP side: the two ethene KEEP canaries carry cited 1,2- parent
        # locants, so their single E/Z descriptor keeps its locant
        # -> `(1Z)-1,2-dibromo-1-chloro-2-iodoethene`, `(1E)-1,2-difluoroethene`.
        # (They never reach the diazene path, but the licence is identical.)
        ("1,2-dibromo-1-chloro-2-iodoethene", True),
        ("1,2-difluoroethene", True),
        # A substituted triazene cites its ene locant -> keep (deferred here).
        ("1,3-diphenyltriaz-1-ene", True),
    ])
    def test_parent_scope_numeral(self, name, has_numeral):
        assert _parent_scope_has_numeral(name) is has_numeral


class TestKeepCanariesUnchanged:
    """The three Phase-10 KEEP canaries must be untouched by the diazene E/Z
    injection: two carry cited parent locants and the third has
    four R/S descriptors multiplicity, `len != 1`), so all three keep
    their descriptor locants. None routes through the polyazane path, but pin the
    full-engine output so a future numeral strip that DID reach them is caught."""

    @pytest.mark.parametrize("smiles,expected", [
        ("Cl/C(Br)=C(\\Br)I", "(1Z)-1,2-dibromo-1-chloro-2-iodoethene"),
        ("F/C=C/F", "(1E)-1,2-difluoroethene"),
        ("C1CC[C@H]2C[C@H]3CCCC[C@H]3C[C@H]2C1",
         "(4aR,8aR,9aS,10aS)-tetradecahydroanthracene"),
        # And the target it must NOT disturb, end to end:
        ("c1ccc(/N=N\\c2ccccc2)cc1", "(Z)-diphenyldiazene"),
    ])
    def test_full_engine(self, smiles, expected):
        assert name_compound(smiles) == expected


class TestFailClosed:
    @pytest.mark.parametrize("smiles,why", [
        ("CN", "methylamine: no N-N bond"),
        ("NCCN", "ethylenediamine: N-C-C-N, no N-N bond"),
        ("NCCCN", "1,3-diaminopropane"),
        ("NO", "hydroxylamine: N-O, not N-N"),
        ("CC(C)=NN", "acetone hydrazone: C=N-N ylidene substituent"),
        ("[N-]=[N+]=N", "azide: charged + two cumulated double bonds"),
        ("c1cc[nH]n1", "pyrazole: N-N in a ring"),
        ("c1ccccc1N", "aniline: single N"),
    ])
    def test_declines(self, smiles, why):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pytest.skip(f"RDKit rejects SMILES ({why})")
        assert name_polyazane(mol) is None, why

    def test_none_input(self):
        assert name_polyazane(None) is None
