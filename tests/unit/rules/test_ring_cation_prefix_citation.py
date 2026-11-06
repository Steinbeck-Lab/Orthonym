"""Ring cations whose prefixes differ only in a stereodescriptor: one name per species (LO1).

``ions._p74_ring_substituent_prefix`` cited the prefixes of one alphanumerical key ('[(1R)-1-chloroethyl]',
'[(1S)-1-chloroethyl]') in the order in which the atoms were met, and ``_emit_ring_cumulative_suffix`` (the
0-H ring cations), ``emit_ring_poly_cation_ium`` and ``azolium_demote_name`` numbered a ring that the heteroatoms,
the 'ium' and the set of prefix locants leave open by the canonical rank of the atoms. So ONE species had a name
for each way its SMILES was written (measured: 211 of 250 panel molecules had two or more names over seven
spellings), and in a symmetric ring ('3,5' of a pyridinium, '3,5' of an N,N-dimethylpiperidinium, '4,5' of a
1,3-dimethylimidazolium) the default tier declined 150 of 1,750 spellings. The record that opened this item read
the output as "S before R" (57 of 57 rows): every one of its SMILES wrote the S side chain first.

What the Blue Book decides (every line opened in ``the Blue Book Blue Book``):

* ``### **** NUMBERING`` (:3219), clause (j) (:3346): "When there is a choice for lower locants related to
  the presence of stereogenic centers or stereoisomers, the lower locant is assigned to CIP stereodescriptors
  *Z*, *R*, *M*, and *r* (pseudoasymmetry) that are preferred to *E*, *S*, *P*, and *s*, respectively". Its
  examples reach the descriptors INSIDE a prefix: '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)'
  (:3399); '1-[(1r,4r)-4-methylcyclohexyl]-2-[(1s,4s)-4-methylcyclohexyl]ethane-1,1,2,2-tetracarbonitrile (PIN)'
  (:3409, "the substituent denoted by the 'r' stereodescriptor receives the lowest locant, '1'"). So a ring
  whose numbering is free ('3,5', '2,6', '4,5' of the symmetric rings below) is numbered with the R prefix at the
  lower locant, and the tie key of the zwitterion emitter (``ring_numbering_tie_key``: (g) then (j)) now decides it
  in these producers too. Clause (g) (:3307, "lowest locants for the substituent cited first as a prefix in the
  name"; '1-methyl-4-nitronaphthalene (PIN)') is part of that key, so a ring whose two different prefixes tie on
  (f) is numbered by (g) and no longer by the canonical rank (the ``G`` table).
* ``### **** ALPHANUMERICAL ORDER`` (:3436),:3446: "In these subsections the principles of alphanumerical
  order do not include Greek letters (except in conjunctive names) or isotopic or stereochemical descriptors."
  The two prefixes tie on it; the Blue Book orders them with ``****`` (:22603): "When names based on
  alphanumerical order and isotopic descriptors are the same, further choice depends on the alphabetic order of
  the stereochemical descriptors 'R' and 'S'" ('1-[(1R)-1-bromoethyl]-1-[(1S)-1-bromoethyl]cyclopentane (PIN)',
  :22609;:22589 "the configurational symbols are compared and 'R' precedes 'S'").

What it does NOT decide, and what is ASSUMED: the order of citation of two such prefixes when the numbering is
FIXED and the R prefix stands at the HIGHER locant (``FIXED_HIGHER`` below: 'pyridin-1-ium' 2,4 with S at 2 and R
at 4). and:22589 read "when the alphanumerical order is the same", and (:22236, "Then, if
still there is a choice, numerical locants are considered in the order of their appearance in the name") makes
'2-[S]-4-[R]' earlier than '4-[R]-2-[S]'; against that, the verbatim PIN of ``### ****`` (:45465)
'(2R,3s,4S,6R)-2,6-dichloro-5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]heptan-4-ol (PIN)' cites R first at the
higher locant, while ```` Example 2 (:45437) '5-[(2E)-but-2-en-1-yl]-9-[(2Z)-but-2-en-1-yl]trideca-2,5,8,11-
tetraen-7-ol (PIN)' cites E first at the lower one. The tests pin R first, as ``prefix_citation_sort_key`` tier 3
does for the neutral-ring producers and as ``emit_zwitterion_ring_carboxylate`` already did
('(3R,5S)-5-[(1R)-...]-3-[(1S)-...]piperidin-1-ium-4-carboxylate', test_ring_zwitterion_stereo.py), so that one
species has one name; those names kept the tier they had (``pin_verified``) and are not claimed to be the PIN
beyond that. The ``FIXED_LOWER`` rows are R first at the lower locant: and the lowest-locant reading agree.

Not changed (measured, see `internal notes`): a ring cation that carries a hydron ('[nH+]',
'[NH2+]') is named by the neutral ring's producer, which has the same free-numbering gap for prefix descriptors
(the default tier declines '3,5-bis(1-chloroethyl)pyridin-1-ium' with one R and one S side chain, as before).

Every name is read back by a FRESH OPSIN call and must give the input's full InChIKey (stereo layers included) and
its fixed-H InChI. Each species is named in three spellings (the SMILES of the panel plus two RDKit random
spellings that gave another name before the change), at the default tier and at the CLI's best-effort tier.
"""
import re

import pytest

from tests.support.jars import jar_or_skip
from tests.support.read_back import same_hydron_sites

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# (j): the numbering is free, the R prefix takes the lower locant (and is cited first)
FREE = [
    (('C[n+]1cc([C@H](C)O)cc([C@@H](C)O)c1', 'C[C@H](O)c1cc([C@@H](C)O)c[n+](C)c1', '[C@H](c1c[n+](cc(c1)[C@H](O)C)C)(O)C'),
     "3-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-1-methylpyridin-1-ium"),
    (('C[C@H](Cl)c1cccc([C@@H](C)Cl)[n+]1C', 'C[n+]1c([C@H](Cl)C)cccc1[C@H](C)Cl', '[C@@H](C)(c1[n+](C)c([C@H](C)Cl)ccc1)Cl'),
     "2-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl]-1-methylpyridin-1-ium"),
    (('C[C@H](O)C1CC([C@@H](C)O)C[N+](C)(C)C1', 'C1C([C@H](O)C)CC([C@@H](O)C)C[N+]1(C)C', 'C1(CC(C[N+](C1)(C)C)[C@@H](O)C)[C@H](O)C'),
     "3-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-1,1-dimethylpiperidin-1-ium"),
    (('C[C@H](Cl)C1CCCC([C@@H](C)Cl)[N+]1(C)C', 'C1CC([C@H](Cl)C)[N+](C)(C)C(C1)[C@@H](Cl)C', 'C1CCC([C@H](Cl)C)[N+](C)(C)C1[C@@H](Cl)C'),
     "2-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl]-1,1-dimethylpiperidin-1-ium"),
    (('C[C@H](Cl)C1C[N+](C)(C)CC1[C@@H](C)Cl', 'C[N+]1(CC([C@@H](Cl)C)C([C@@H](C)Cl)C1)C', 'C[C@@H](Cl)C1C(C[N+](C)(C1)C)[C@H](C)Cl'),
     "3-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl]-1,1-dimethylpyrrolidin-1-ium"),
    (('C[C@H](Cl)C1CCC([C@@H](C)Cl)[N+]1(C)C', 'C1CC([C@H](C)Cl)[N+](C1[C@@H](C)Cl)(C)C', 'C1(CCC([N+]1(C)C)[C@H](C)Cl)[C@H](Cl)C'),
     "2-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1,1-dimethylpyrrolidin-1-ium"),
    (('C[C@H](Cl)C1COCC([C@@H](C)Cl)[N+]1(C)C', '[N+]1(C)(C)C(COCC1[C@@H](Cl)C)[C@@H](C)Cl', 'Cl[C@@H](C1[N+](C)(C([C@H](C)Cl)COC1)C)C'),
     "3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-4,4-dimethylmorpholin-4-ium"),
    (('C[C@H](Cl)c1c[n+](C)cc([C@@H](C)Cl)n1', 'Cl[C@@H](c1c[n+](C)cc([C@H](C)Cl)n1)C', 'c1(c[n+](C)cc([C@H](Cl)C)n1)[C@@H](Cl)C'),
     "3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1-methylpyrazin-1-ium"),
    (('C[C@H](O)c1c([C@@H](C)O)[n+](C)cn1C', '[C@H](c1c([C@H](O)C)[n+](C)cn1C)(O)C', 'O[C@@H](C)c1n(C)c[n+](C)c1[C@H](O)C'),
     "4-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-1,3-dimethyl-1H-imidazol-3-ium"),
    (('Cc1c([C@H](C)Cl)c[n+](C)cc1[C@@H](C)Cl', 'c1([C@@H](Cl)C)c(c([C@H](Cl)C)c[n+](c1)C)C', 'c1([C@H](Cl)C)c(C)c([C@@H](Cl)C)c[n+](C)c1'),
     "3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1,4-dimethylpyridin-1-ium"),
    (('C[C@H](O)C1CC([C@@H](C)O)C[S+](C)C1', 'C[C@H](O)C1CC(C[S+](C)C1)[C@H](O)C', 'C1(C[S+](C)CC(C1)[C@@H](O)C)[C@H](O)C'),
     "3-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-1-methylthian-1-ium"),
    (('C[C@H](N)C1CC([C@@H](C)N)C[P+](C)(C)C1', '[P+]1(C)(CC([C@@H](N)C)CC(C1)[C@H](N)C)C', 'C[C@@H](C1C[P+](C)(CC(C1)[C@@H](C)N)C)N'),
     "3-[(1R)-1-aminoethyl]-5-[(1S)-1-aminoethyl]-1,1-dimethylphosphinan-1-ium"),
    (('C[C@H](Cl)C1C[NH+](C)C([C@@H](C)Cl)C[NH+]1C', '[NH+]1(C(C[NH+](C(C1)[C@H](Cl)C)C)[C@H](C)Cl)C', '[C@@H](C)(C1C[NH+](C([C@H](C)Cl)C[NH+]1C)C)Cl'),
     "2-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1,4-dimethylpiperazine-1,4-diium"),
    (('C[C@H](Cl)C1C[NH+](C)CC([C@@H](C)Cl)[NH+]1C', 'Cl[C@H](C1[NH+](C([C@@H](C)Cl)C[NH+](C1)C)C)C', 'C[NH+]1CC([NH+](C(C1)[C@@H](Cl)C)C)[C@@H](C)Cl'),
     "2-[(1R)-1-chloroethyl]-6-[(1S)-1-chloroethyl]-1,4-dimethylpiperazine-1,4-diium"),
    (('C[C@H](Cl)C1C([C@@H](C)Cl)[NH+](C)CC[NH+]1C', '[NH+]1(C)C([C@@H](C)Cl)C([C@@H](Cl)C)[NH+](CC1)C', 'C1[NH+](C([C@H](C)Cl)C([NH+](C)C1)[C@H](Cl)C)C'),
     "2-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]-1,4-dimethylpiperazine-1,4-diium"),
    (('CC[C@@H](C)c1cc([C@@H](C)CC)c[n+](C)c1', 'c1[n+](C)cc(cc1[C@H](C)CC)[C@@H](C)CC', 'C[C@@H](CC)c1c[n+](C)cc([C@H](C)CC)c1'),
     "3-[(2R)-butan-2-yl]-5-[(2S)-butan-2-yl]-1-methylpyridin-1-ium"),
]

# fixed numbering, the R prefix at the lower locant: and the lowest-locant reading agree
FIXED_LOWER = [
    (('C[C@H](O)c1sc[n+](C)c1[C@@H](C)O', 'C[C@H](O)c1sc[n+](c1[C@@H](C)O)C', 'C[C@H](c1c([C@@H](O)C)sc[n+]1C)O'),
     "4-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-3-methyl-1,3-thiazol-3-ium"),
    (('C[C@H](Cl)C1=C[NH+](C)CC([C@@H](C)Cl)=C1', 'C1(=C[NH+](C)CC(=C1)[C@@H](C)Cl)[C@@H](Cl)C', '[NH+]1(C)C=C([C@H](C)Cl)C=C([C@@H](C)Cl)C1'),
     "3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1-methyl-1,2-dihydropyridin-1-ium"),
    (('C[C@H](Cl)c1cc[n+](C)c([C@@H](C)Cl)c1', 'c1c(cc([C@H](Cl)C)[n+](C)c1)[C@H](C)Cl', 'c1(cc[n+](c(c1)[C@@H](C)Cl)C)[C@@H](Cl)C'),
     "2-[(1R)-1-chloroethyl]-4-[(1S)-1-chloroethyl]-1-methylpyridin-1-ium"),
    (('C[C@H](O)c1ccc[n+](C)c1[C@@H](C)O', '[n+]1(C)cccc(c1[C@@H](C)O)[C@@H](O)C', '[C@H](c1c([n+](ccc1)C)[C@@H](C)O)(O)C'),
     "2-[(1R)-1-hydroxyethyl]-3-[(1S)-1-hydroxyethyl]-1-methylpyridin-1-ium"),
    (('CC[C@@H](C)c1cc([C@@H](C)CC)cc[n+]1C', 'c1cc([C@H](CC)C)cc([n+]1C)[C@H](C)CC', 'CC[C@H](C)c1cc[n+](C)c(c1)[C@@H](CC)C'),
     "2-[(2R)-butan-2-yl]-4-[(2S)-butan-2-yl]-1-methylpyridin-1-ium"),
    (('C[C@H](N)c1cn(C)c([C@@H](C)N)[n+]1C', '[C@@H](c1cn(c([C@@H](C)N)[n+]1C)C)(C)N', '[n+]1(C)c(cn(c1[C@H](N)C)C)[C@@H](N)C'),
     "2-[(1R)-1-aminoethyl]-4-[(1S)-1-aminoethyl]-1,3-dimethyl-1H-imidazol-3-ium"),
]

# fixed numbering, the R prefix at the HIGHER locant: R cited first (ASSUMED, see the docstring)
FIXED_HIGHER = [
    (('C[C@H](O)c1c([C@@H](C)O)sc[n+]1C', 'c1[n+](c([C@@H](O)C)c([C@@H](C)O)s1)C', 'c1([C@H](C)O)c(sc[n+]1C)[C@H](O)C'),
     "5-[(1R)-1-hydroxyethyl]-4-[(1S)-1-hydroxyethyl]-3-methyl-1,3-thiazol-3-ium"),
    (('C[C@H](Cl)C1=C(C(=O)[O-])C([C@@H](C)Cl)=C[NH+](C)C1', 'C1C(=C(C([C@@H](C)Cl)=C[NH+]1C)C([O-])=O)[C@@H](Cl)C', 'C1[NH+](C)CC([C@@H](Cl)C)=C(C=1[C@@H](C)Cl)C([O-])=O'),
     "5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]-1-methyl-1,2-dihydropyridin-1-ium-4-carboxylate"),
    (('C[C@H](O)c1c([C@@H](C)O)ccc[n+]1C', '[C@H](c1c(ccc[n+]1C)[C@H](O)C)(O)C', 'c1cc[n+](c([C@@H](O)C)c1[C@H](O)C)C'),
     "3-[(1R)-1-hydroxyethyl]-2-[(1S)-1-hydroxyethyl]-1-methylpyridin-1-ium"),
    (('CC[C@@H](C)c1cc[n+](C)c([C@@H](C)CC)c1', 'CC[C@H](C)c1cc([C@@H](CC)C)cc[n+]1C', 'CC[C@@H](c1cc([C@@H](CC)C)cc[n+]1C)C'),
     "4-[(2R)-butan-2-yl]-2-[(2S)-butan-2-yl]-1-methylpyridin-1-ium"),
    (('C[C@H](Cl)c1cc([C@@H](C)Cl)cc[n+]1C', '[C@@H](Cl)(c1[n+](C)ccc([C@@H](C)Cl)c1)C', 'C[C@@H](c1[n+](ccc([C@@H](C)Cl)c1)C)Cl'),
     "4-[(1R)-1-chloroethyl]-2-[(1S)-1-chloroethyl]-1-methylpyridin-1-ium"),
    (('C[C@H](Cl)C1=CC([C@@H](C)Cl)=C[NH+](C)C1', 'C1(C[NH+](C)C=C([C@@H](C)Cl)C=1)[C@@H](Cl)C', 'C[C@H](Cl)C1C[NH+](C)C=C([C@H](Cl)C)C=1'),
     "5-[(1R)-1-chloroethyl]-3-[(1S)-1-chloroethyl]-1-methyl-1,2-dihydropyridin-1-ium"),
    (('C/C=C\\c1cc[n+](C)c(/C=C/C)c1', 'c1(/C=C/C)cc(cc[n+]1C)/C=C\\C', 'c1([n+](C)ccc(/C=C\\C)c1)/C=C/C'),
     "1-methyl-4-[(1Z)-prop-1-en-1-yl]-2-[(1E)-prop-1-en-1-yl]pyridin-1-ium"),
]

# (g): two different prefixes tie on (f); the one cited first takes the lower locant (no stereo, or one centre)
G = [
    (('Cc1cc(Cl)c[n+](C)c1', 'Cc1cc(c[n+](C)c1)Cl', 'Cc1c[n+](C)cc(Cl)c1'),
     "3-chloro-1,5-dimethylpyridin-1-ium"),
    (('C[n+]1c(Cl)cccc1Br', 'Brc1cccc(Cl)[n+]1C', '[n+]1(c(Br)cccc1Cl)C'),
     "2-bromo-6-chloro-1-methylpyridin-1-ium"),
    (('C[N+]1(C)CC(Cl)CC(Br)C1', 'C1(Cl)CC(Br)C[N+](C)(C1)C', 'C[N+]1(CC(CC(Br)C1)Cl)C'),
     "3-bromo-5-chloro-1,1-dimethylpiperidin-1-ium"),
    (('CC1CCCC(Cl)[P+]1(C)C', 'C1C(Cl)[P+](C)(C(C)CC1)C', 'C1(Cl)CCCC(C)[P+]1(C)C'),
     "2-chloro-1,1,6-trimethylphosphinan-1-ium"),
    (('CC1COCC(O)[N+]1(C)C', 'C1C(C)[N+](C)(C(O)CO1)C', 'C1OCC(O)[N+](C)(C1C)C'),
     "3-hydroxy-4,4,5-trimethylmorpholin-4-ium"),
    (('CC1CC(N)C[S+](C)C1', 'C[S+]1CC(C)CC(N)C1', 'CC1C[S+](C)CC(N)C1'),
     "3-amino-1,5-dimethylthian-1-ium"),
    (('C[C@H](Cl)c1cc(C)c[n+](C)c1', 'c1(C)c[n+](cc(c1)[C@@H](Cl)C)C', '[C@@H](Cl)(c1cc(C)c[n+](c1)C)C'),
     "3-[(1S)-1-chloroethyl]-1,5-dimethylpyridin-1-ium"),
]

# controls: prefixes with the SAME descriptor are cited as one group, as before
SAME_DESCRIPTOR = [
    (("C[C@H](Cl)c1cc([C@H](C)Cl)c[n+](C)c1",), "3,5-bis[(1S)-1-chloroethyl]-1-methylpyridin-1-ium"),
    (("C[C@@H](Cl)c1cc([C@@H](C)Cl)c[n+](C)c1",), "3,5-bis[(1R)-1-chloroethyl]-1-methylpyridin-1-ium"),
    (("C[C@H](Cl)C1CC([C@H](C)Cl)C[N+](C)(C)C1",), "3,5-bis[(1S)-1-chloroethyl]-1,1-dimethylpiperidin-1-ium"),
    (("C[C@H](Cl)c1c([C@H](C)Cl)[n+](C)cn1C",), "4,5-bis[(1S)-1-chloroethyl]-1,3-dimethyl-1H-imidazol-3-ium"),
]


def _named_in_every_spelling(spellings, name):
    jar_or_skip()
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    namers = (Orthonym(style="pin"), Orthonym(style="pin", **_emit_tier_flags("best-effort")))
    for smiles in spellings:
        for namer in namers:
            r = namer.name_tiered(smiles)
            assert (r["name"], r["tier"]) == (name, "pin_verified"), (smiles, r)
    assert same_hydron_sites(name, spellings[0]), name


def _cited_locants(name):
    """The locants of the stereo-descriptor prefixes of ``name``, in the order they are cited."""
    return [int(x) for x in re.findall(r"(?:^|-)(\d+)-\[\(", name)]


@pytest.mark.parametrize("spellings,name", FREE, ids=[n for _, n in FREE])
def test_free_numbering_gives_the_r_prefix_the_lower_locant(spellings, name):
    """ (j),:3346 and the example:3399: the ring's numbering is a choice, and R takes the lower locant."""
    locants = _cited_locants(name)
    assert len(locants) == 2 and locants == sorted(locants), name
    _named_in_every_spelling(spellings, name)


@pytest.mark.parametrize("spellings,name", FIXED_LOWER, ids=[n for _, n in FIXED_LOWER])
def test_fixed_numbering_r_prefix_at_the_lower_locant_is_cited_first(spellings, name):
    locants = _cited_locants(name)
    assert len(locants) == 2 and locants == sorted(locants), name
    _named_in_every_spelling(spellings, name)


@pytest.mark.parametrize("spellings,name", FIXED_HIGHER, ids=[n for _, n in FIXED_HIGHER])
def test_fixed_numbering_r_prefix_at_the_higher_locant_is_cited_first_assumed(spellings, name):
    """ASSUMED (docstring): the Blue Book has a (PIN) for each order. One species, one name: R (Z) first."""
    locants = _cited_locants(name)
    assert len(locants) == 2 and locants != sorted(locants), name
    _named_in_every_spelling(spellings, name)


@pytest.mark.parametrize("spellings,name", G, ids=[n for _, n in G])
def test_prefix_cited_first_takes_the_lower_locant_where_the_ring_numbering_is_open(spellings, name):
    """ (g),:3307: 'chloro' is cited before 'methyl', so it takes the lower of the two open locants."""
    _named_in_every_spelling(spellings, name)


@pytest.mark.parametrize("spellings,name", SAME_DESCRIPTOR, ids=[n for _, n in SAME_DESCRIPTOR])
def test_prefixes_with_one_descriptor_are_still_one_group(spellings, name):
    _named_in_every_spelling(spellings, name)
