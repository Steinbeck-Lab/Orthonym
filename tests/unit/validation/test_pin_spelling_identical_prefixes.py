""": a substituent cited twice in one scope is one multiplied prefix (SL lane G, G1).

A round trip proves the molecule, never the spelling: OPSIN reads '2-(4-methoxyphenyl)-2-methyl-N-
methylpropanamide' back to the structure of the preferred '2-(4-methoxyphenyl)-N,2-dimethyl-
propanamide'. "MULTIPLICATIVE PREFIXES 'DI', 'TRI', ETC. vs. 'BIS', 'TRIS', ETC."
(the Blue Book), (:7027): "Multiplicative prefixes... are the principal method for
describing a multiplicity of identical features of a structure in chemical names"; the locants of
the group are cited together, (:3195) "Italic capital and lower-case letter locants are
lower than Greek letter locants, which, in turn, are lower than numerals". The Blue Book's own
names: 'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)' (:32879), 'N,N,2-trimethyl-3-{...}propanamide
(PIN)' (:21624). The check reads the name text only.
"""
import pytest
from rdkit import Chem

from orthonym.validation import pin_spelling as ps

pytestmark = pytest.mark.unit

MOL = Chem.MolFromSmiles("C")


def _rules(name):
    return [f.rule for f in ps.check_pin_spelling(MOL, name)]


@pytest.mark.parametrize("name", [
    "2-(4-methoxyphenyl)-2-methyl-N-methylpropanamide",
    "1-(1-benzofuran-2-yl)-2-methoxy-3,3-dimethyl-N-methylbutan-1-amine",
    "3-hydrazinyl-2-methyl-N,N-dimethyldecan-2-amine",
    "(2S,3S)-3-hydroxy-2-methyl-N,N-dimethyl-3-(4-methylphenyl)propanamide",
    "2,6-diethyl-N-ethyl-4-methylaniline",
    "4-methyl-N-methylaniline",
    "(2S,3R)-2-hydroxy-3-hydroxy-2-(1-hydroxytridecyl)-4-methoxycyclopentan-1-one",
    "2,3,4-trihydroxy-5-hydroxy-5-[(phosphonooxy)methyl]cyclohexan-1-one",
    "3-(4-methyl-N-methylanilino)propanenitrile",             # the same words inside an enclosure
    "N-methyl-3,3-dimethyl-2-propanamidobutanamide",
    "2-(4-methoxyphenyl)-N-(4-methoxyphenyl)acetamide",       # an enclosed prefix twice
    "2-chloro-4-chloropyridine",
])
def test_a_substituent_cited_twice_in_one_scope_fails(name):
    assert "P-16.3.1" in _rules(name)


@pytest.mark.parametrize("name", [
    # the Blue Book's own names
    "N,4-dimethyl-N-(3-methylphenyl)benzamide",                                   #:32879
    "N,N,2-trimethyl-3-{4-methyl-3-[2-methyl-3-(methylamino)-3-oxopropyl]phenyl}propanamide",  #:21624
    "N,N,N,1-tetramethylquinolin-1-ium-3-aminium",                                #:42213
    "4-[(2-hydroxyethyl)amino]-N-{2-[(2-hydroxyethyl)amino]ethyl}butanamide",     #:33574, two scopes
    # the multiplied forms the engine writes
    "2-(4-methoxyphenyl)-N,2-dimethylpropanamide",
    "N,N,2-trimethylpropan-2-amine",
    "2,3-dihydroxy-2-(1-hydroxytridecyl)-4-methoxycyclopentan-1-one",
    "N,2,6-triethyl-4-methylaniline",
    "N,2-bis(4-methoxyphenyl)acetamide",
    # one word, two substituents: tert-butyl is not butyl; two configurations are two groups
    "4-butyl-2-tert-butylphenol",
    "2-[(1R)-1-hydroxyethyl]-3-[(1S)-1-hydroxyethyl]pyridine",
    # a locanted word that is the first part of a compound word is not the simple prefix
    "8-hydroxy-4-methoxy-5-methoxycarbonyl-3-methyl-3,4-dihydro-1H-2-benzopyran-1-one",
    # an isotopically modified prefix is another substituent,:43718;:43764)
    "2-(13C)methyl-3-methylpyridine",
    "(2R)-1-(131I)iodo-3-iodopropan-2-ol",
    # the same word in two scopes
    "2-(2-chloroethyl)-3-chloropyridine",
    "1-chloro-2-(chloromethyl)benzene",
])
def test_names_without_the_defect_pass(name):
    assert "P-16.3.1" not in _rules(name)


def test_the_message_names_the_substituent():
    failure = next(f for f in ps.check_pin_spelling(MOL, "4-methyl-N-methylaniline") if f.rule == "P-16.3.1")
    assert "'methyl' is cited 2 times in one scope" in failure.detail
