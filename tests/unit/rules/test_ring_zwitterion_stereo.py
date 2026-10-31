"""A zwitterion whose ring cation and ring carboxylate share a SATURATED ring (proline, hydroxyproline,
pipecolate, piperazine-2-carboxylate, thiazolidine-4-carboxylate, morpholine carboxylates, the N-methyl
and N,N-dimethyl betaines) is named with its ring stereodescriptors, in the numbering of the name.

``emit_zwitterion_ring_carboxylate`` built '<prefixes><ring>-<n>-ium-<m>-carboxylate' and cited no
descriptor, so every such zwitterion WITH a stereocentre in the ring had its name withdrawn by the stereo
backstop at the default tier ('unknown organic compound'); the best-effort tier added the descriptors and
labelled the name ``pin_unverified``. The descriptors are now cited by the emitter itself.

What the Blue Book says (every line opened in ``the Blue Book Blue Book``):

* "Zwitterionic compounds with at least one ionic center on a characteristic group" (:42445),
  :42447: "Zwitterionic compounds with at least one ionic center on a characteristic group may be named by
  adding the appropriate ionic suffix to the name of the ionic parent hydride. In names, cationic suffixes
  are cited before anionic suffixes. For assignment of lower locants, ionic centers on skeletal atoms of
  the parent hydride are preferred to the locants for positions of attachment of characteristic groups
  denoted by ionic suffixes."; example '1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate (PIN)' (:42456).
  The ring nitrogen of proline is a skeletal atom of the parent hydride pyrrolidine and the carboxylate is
  on the same ring: the ionic centers are on ONE parent compound, so the PIN is
  '(2S)-pyrrolidin-1-ium-2-carboxylate' ('ium' at 1, the carboxylate at 2), not an 'azaniumyl' prefix on an
  anionic parent "Anionic and cationic centers on different parent structures",:42462, sentence
  :42464, is the acyclic amino acid: 'azaniumylacetate').
* "Ionic centers in the same parent structure" (:42415),:42419: "For nomenclature purposes,
  zwitterionic compounds having the ionic centers in the same parent structure are not considered as
  neutral compounds."
* (:54554): "Zwitterionic amino acids are named in two ways: (1) by prefixing the name of the
  cationic substituent group into that of the anionic parent (see and using CIP stereodescriptors
  to describe the configuration; (2) by adding the term 'zwitterion' to the name of the amino acid."
  Neither is marked (PIN). Method (1) is the systematic one, with CIP descriptors, as here. Example (3S)-3-
  hydroxy-L-proline / (2S,3S)-3-hydroxyproline "Use of CIP stereodescriptors",:54312,
  example:54318): the descriptors are CIP.
* "NAMING OF STEREOISOMERS" (:44639),:44643: "They are placed at the front of the complete name
  when related to the parent structure"; the locant in each is that of the numbering of the name.
* "NUMBERING" (:3219): when the ring has a choice of numbering that the 'ium', the carboxylate and the
  set of prefix locants do not settle, (g) (:3307) "lowest locants for the substituent cited first as a prefix
  in the name" ('1-methyl-4-nitronaphthalene (PIN)') decides, then (j) (:3346) "When there is a choice for
  lower locants related to the presence of stereogenic centers or stereoisomers, the lower locant is assigned
  to CIP stereodescriptors Z, R, M, and r (pseudoasymmetry) that are preferred to E, S, P, and s" ('(2R,4S)-2,4-
  difluoropentane (PIN)',:3383): the meso '3,5-dimethylpiperidin-1-ium-4-carboxylate' is '(3R,5S)-'.
* A quaternary ring nitrogen ('[N@+]', no hydrogen) is a stereocentre of the name and is cited at its locant
  ('(1R,2S)-1-ethyl-1-methylpyrrolidin-1-ium-2-carboxylate'); a fresh OPSIN call reads that name back to the
  input, descriptor at nitrogen included. Only a stereogenic ring N-H ('[N@H+]') is declined: OPSIN does not
  parse '(1R,2R)-1-methylazetidin-1-ium-2-carboxylate'.

NOT changed, and pinned below as controls: the L-proline zwitterion keeps the retained name 'L-proline'
(``rules.salts.RETAINED_AMINO_ACID_ZWITTERIONS``,,:54518: "It is nevertheless convenient to draw
the conventional form... and to name the amino acid alanine as 2-aminopropanoic acid rather than as
2-azaniumylpropanoate"); it had a name and a tier before, and a name that had a PIN is not withdrawn.

Every name is read back by a FRESH OPSIN call and must give the input's InChIKey (stereo layers included), its
fixed-H InChI (the site of the hydron) and a stereo-aware graph comparison (every tetrahedral centre of the input,
the ring nitrogen too, carried with the same configuration, and none that the input lacks: the InChIKey alone does
not see every centre).
"""
import pytest

from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _fixed_h(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    return inchi.MolToInchi(Chem.MolFromSmiles(smi), options="/FixedH /SNon")


def _key(smi):
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


def _stereo_aware(smiles, parsed, partial=False):
    """Every tetrahedral centre and E/Z bond of ``smiles`` is carried with the same configuration by ``parsed``,
    and ``parsed`` carries none that ``smiles`` lacks (a graph match without charges, bond orders or hydrogens).
    ``partial``: ``parsed`` may carry fewer (OPSIN reads no 'r'/'s'), but none that conflicts."""
    from rdkit import Chem

    from orthonym.perception.charge_shift import stereo_elements

    def generic(m):
        rw = Chem.RWMol(Chem.RemoveHs(m))
        for a in rw.GetAtoms():
            a.SetFormalCharge(0), a.SetIsAromatic(False), a.SetNoImplicit(True), a.SetNumExplicitHs(0)
        for b in rw.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE), b.SetIsAromatic(False)
        g = rw.GetMol()
        g.UpdatePropertyCache(False)
        return g

    inp, out = Chem.RemoveHs(Chem.MolFromSmiles(smiles)), Chem.RemoveHs(Chem.MolFromSmiles(parsed))
    want = stereo_elements(inp)
    for match in generic(inp).GetSubstructMatches(generic(out), uniquify=False, maxMatches=20000,
                                                  useChirality=False):
        got = stereo_elements(out, match)
        if got == want or (partial and all(k in want and want[k] == v for k, v in got.items())):
            return True
    return False


def _exact(name, smiles):
    from tests.support.rt_assert import _independent_parse
    parsed = _independent_parse(name)
    return (bool(parsed) and _key(parsed) == _key(smiles) and _fixed_h(parsed) == _fixed_h(smiles)
            and _stereo_aware(smiles, parsed))


def _named(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r
    assert _exact(name, smiles)


def _named_at_both_tiers(smiles, name):
    """The default tier and the general-engine tiers (the CLI's best-effort, and the 'complete' flags this lane
    measured as 'best-effort') give the same name, `pin_verified` at each, read back exactly."""
    jar_or_skip()
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    for namer in (Orthonym(style="pin"), Orthonym(style="pin", **_emit_tier_flags("best-effort")),
                  Orthonym(style="pin", general_fallback=True, allow_aromatic_general=True)):
        r = namer.name_tiered(smiles)
        assert r["name"] == name and r["tier"] == "pin_verified", r
    assert _exact(name, smiles)


@pytest.mark.parametrize("smiles,name", [
    # pyrrolidine: D-proline (L-proline keeps its retained name, see the controls), pyrrolidine-2-carboxylate
    ("O=C([O-])[C@H]1CCC[NH2+]1", "(2R)-pyrrolidin-1-ium-2-carboxylate"),
    # hydroxyprolines: the descriptors of both centres, in the numbering of the name (N 1, carboxylate 2)
    ("O=C([O-])[C@@H]1C[C@@H](O)C[NH2+]1", "(2S,4R)-4-hydroxypyrrolidin-1-ium-2-carboxylate"),
    ("O=C([O-])[C@@H]1C[C@H](O)C[NH2+]1", "(2S,4S)-4-hydroxypyrrolidin-1-ium-2-carboxylate"),
    ("O=C([O-])[C@H]1C[C@@H](O)C[NH2+]1", "(2R,4R)-4-hydroxypyrrolidin-1-ium-2-carboxylate"),
    ("O=C([O-])[C@H]1[NH2+]CC[C@@H]1O", "(2S,3S)-3-hydroxypyrrolidin-1-ium-2-carboxylate"),
    ("O=C([O-])[C@H]1[NH2+]CC[C@H]1O", "(2S,3R)-3-hydroxypyrrolidin-1-ium-2-carboxylate"),
    ("O=C([O-])[C@@H]1[NH2+]CC[C@@H]1O", "(2R,3S)-3-hydroxypyrrolidin-1-ium-2-carboxylate"),
    # the same species drawn from the other end of the SMILES: the name does not depend on the drawing
    ("O[C@@H]1C[C@H]([NH2+]C1)C(=O)[O-]", "(2S,4R)-4-hydroxypyrrolidin-1-ium-2-carboxylate"),
])
def test_proline_type_zwitterions_cite_their_ring_stereodescriptors(smiles, name):
    _named(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    # piperidine: pipecolate (2) and nipecotate (3); the carboxylate takes the lowest locant it can
    ("O=C([O-])[C@@H]1CCCC[NH2+]1", "(2S)-piperidin-1-ium-2-carboxylate"),
    ("O=C([O-])[C@H]1CCCC[NH2+]1", "(2R)-piperidin-1-ium-2-carboxylate"),
    ("O=C([O-])[C@@H]1CCC[NH2+]C1", "(3R)-piperidin-1-ium-3-carboxylate"),
    # piperazine-2-carboxylate: both nitrogens skeletal, the 'ium' on N-1, the carboxylate beside it
    ("O=C([O-])[C@H]1CNCC[NH2+]1", "(2R)-piperazin-1-ium-2-carboxylate"),
    ("O=C([O-])[C@@H]1CNCC[NH2+]1", "(2S)-piperazin-1-ium-2-carboxylate"),
    # thiazolidine-4-carboxylate (thioproline): the heteroatom locants are in the stem (S-1, N-3)
    ("O=C([O-])[C@@H]1CSC[NH2+]1", "(4R)-1,3-thiazolidin-3-ium-4-carboxylate"),
    ("O=C([O-])[C@H]1CSC[NH2+]1", "(4S)-1,3-thiazolidin-3-ium-4-carboxylate"),
])
def test_other_saturated_rings_cite_their_ring_stereodescriptors(smiles, name):
    _named(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    # a substituent on the 'ium' nitrogen is a prefix at its locant, '1-methylpyridin-1-ium (PIN)',
    #:41394); the descriptors still stand at the front of the complete name
    ("C[NH+]1CCC[C@H]1C(=O)[O-]", "(2S)-1-methylpyrrolidin-1-ium-2-carboxylate"),
    ("C[N+]1(C)CCC[C@H]1C(=O)[O-]", "(2S)-1,1-dimethylpyrrolidin-1-ium-2-carboxylate"),
    ("C[N+]1(C)CCC[C@@H]1C(=O)[O-]", "(2R)-1,1-dimethylpyrrolidin-1-ium-2-carboxylate"),
    ("C[N+]1(C)C[C@H](O)C[C@H]1C(=O)[O-]", "(2S,4R)-4-hydroxy-1,1-dimethylpyrrolidin-1-ium-2-carboxylate"),
    ("C[N+]1(C)CC[C@H](O)[C@@H]1C(=O)[O-]", "(2R,3S)-3-hydroxy-1,1-dimethylpyrrolidin-1-ium-2-carboxylate"),
    ("C[N+]1(C)CCCC[C@H]1C(=O)[O-]", "(2S)-1,1-dimethylpiperidin-1-ium-2-carboxylate"),
])
def test_n_substituted_betaines_cite_their_ring_stereodescriptors(smiles, name):
    _named(smiles, name)


@pytest.mark.parametrize("smiles,name", [
    # a ring without a stereocentre has no descriptor to cite: byte-identical to before
    ("O=C([O-])C1CCC[NH2+]1", "pyrrolidin-1-ium-2-carboxylate"),
    ("O=C([O-])C1CC(O)C[NH2+]1", "4-hydroxypyrrolidin-1-ium-2-carboxylate"),
    ("O=C([O-])C1CSC[NH2+]1", "1,3-thiazolidin-3-ium-4-carboxylate"),
    ("O=C([O-])C1CSCC[NH2+]1", "thiomorpholin-4-ium-3-carboxylate"),
    ("C[n+]1cccc(C(=O)[O-])c1", "1-methylpyridin-1-ium-3-carboxylate"),
    ("C[N+]1(C)CCCCC1C(=O)[O-]", "1,1-dimethylpiperidin-1-ium-2-carboxylate"),
    # the Blue Book's own worked PIN example,:42456
    ("C[n+]1c(C(=O)[O-])cc(-c2ccccc2)cc1-c1ccccc1", "1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate"),
])
def test_names_that_existed_are_unchanged(smiles, name):
    _named(smiles, name)


def test_l_proline_zwitterion_keeps_its_retained_name():
    """The retained name of (:54518) stands for the L-proline zwitterion: it had a name and a tier
    before this change, and a name that had a PIN is not withdrawn. OPSIN reads 'L-proline' as the neutral acid:
    the standard InChIKey (stereo included) is the input's, the fixed-H InChI is not (the hydron is on the
    carboxylate), which is the amino-acid case that ``validation.protonation_identity`` accepts."""
    jar_or_skip()
    from orthonym import Orthonym
    from tests.support.rt_assert import _independent_parse
    smiles = "O=C([O-])[C@@H]1CCC[NH2+]1"
    #:50943 gives no PIN in Chapter (relabel 2026-10-31, internal notes): the name stands at the best-effort tier, labelled below the PIN
    from orthonym.cli import _emit_tier_flags
    r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert r["name"] == "L-proline" and r["tier"] == "systematic_verified", r
    assert _key(_independent_parse("L-proline")) == _key(smiles)


def test_the_emitter_cites_the_descriptors_in_its_own_numbering():
    """``emit_zwitterion_ring_carboxylate`` itself returns the descriptors: they are not added to its
    result afterwards. The locant of each is the locant of the same atom in the name's numbering."""
    from rdkit import Chem

    from orthonym.rules.ions import emit_zwitterion_ring_carboxylate
    smiles = "O=C([O-])[C@@H]1C[C@@H](O)C[NH2+]1"
    mol = Chem.MolFromSmiles(smiles)
    cation = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0][0]
    anion = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() < 0][0]
    assert emit_zwitterion_ring_carboxylate(mol, cation, anion) == "(2S,4R)-4-hydroxypyrrolidin-1-ium-2-carboxylate"


def _emit(smiles):
    from rdkit import Chem

    from orthonym.rules.ions import emit_zwitterion_ring_carboxylate
    mol = Chem.MolFromSmiles(smiles)
    cation = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() > 0][0]
    anion = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() < 0][0]
    return emit_zwitterion_ring_carboxylate(mol, cation, anion)


def _ring_map(mol):
    return {a.GetIdx(): a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}


def test_a_stereogenic_ring_n_h_is_not_cited():
    """A specified '[N@H+]' is a stereocentre of the ring that carries a hydrogen, and OPSIN does not parse a
    descriptor there ('(1R,2R)-1-methylazetidin-1-ium-2-carboxylate' gives no structure), so the stereo-aware
    read-back that every name faces could not confirm it. The emitter declines, as it did before the descriptors
    were cited; the drawing without a specified nitrogen is named (the nitrogen is not a stereocentre of the
    name then: the hydron and the methyl exchange). A QUATERNARY '[N@+]' holds no hydrogen, OPSIN reads its
    descriptor, and it is cited (the tests below)."""
    from rdkit import Chem

    from orthonym.rules.ring_cation_hydro import ring_stereo_prefix
    assert _emit("O=C([O-])[C@H]1CC[N@H+]1C") == ""
    assert _emit("O=C([O-])[C@H]1CC[NH+]1C") == "(2R)-1-methylazetidin-1-ium-2-carboxylate"
    ring = Chem.MolFromSmiles("O=C([O-])[C@H]1CC[N@H+]1C")
    assert ring_stereo_prefix(ring, _ring_map(ring)) is None
    plain = Chem.MolFromSmiles("O=C([O-])C1CCC[NH2+]1")
    assert ring_stereo_prefix(plain, _ring_map(plain)) == ""
    quaternary = Chem.MolFromSmiles("CC[N@+]1(C)CCC[C@H]1C(=O)[O-]")
    assert ring_stereo_prefix(quaternary, _ring_map(quaternary)).endswith("S)-")


@pytest.mark.parametrize("smiles,name", [
    # all four configurations of a pyrrolidinium whose nitrogen (ethyl, methyl, two ring bonds) is a stereocentre
    ("CC[N@+]1(C)CCC[C@H]1C(=O)[O-]", "(1R,2S)-1-ethyl-1-methylpyrrolidin-1-ium-2-carboxylate"),
    ("CC[N@@+]1(C)CCC[C@H]1C(=O)[O-]", "(1S,2S)-1-ethyl-1-methylpyrrolidin-1-ium-2-carboxylate"),
    ("CC[N@+]1(C)CCC[C@@H]1C(=O)[O-]", "(1R,2R)-1-ethyl-1-methylpyrrolidin-1-ium-2-carboxylate"),
    ("CC[N@@+]1(C)CCC[C@@H]1C(=O)[O-]", "(1S,2R)-1-ethyl-1-methylpyrrolidin-1-ium-2-carboxylate"),
    ("CC[N@+]1(C)CCCC[C@H]1C(=O)[O-]", "(1R,2S)-1-ethyl-1-methylpiperidin-1-ium-2-carboxylate"),
    ("CC[N@@+]1(C)CCCC[C@H]1C(=O)[O-]", "(1S,2S)-1-ethyl-1-methylpiperidin-1-ium-2-carboxylate"),
])
def test_a_quaternary_stereogenic_ring_nitrogen_is_cited_at_both_tiers(smiles, name):
    """The decline at a ring nitrogen is for a stereogenic N-H only. A '[N@+]' without a hydrogen is cited in
    the front block with its ring locant,:44643), and every one of the four configurations is read back
    by a fresh OPSIN call to the input, the descriptor at nitrogen included (the stereo-aware comparison would
    fail on '(1S,2S)' for the '[N@+]' drawing). The default tier names it `pin_verified` (it abstained), and
    the general-engine tiers keep the PIN-form name they gave, not the systematic one with the carboxylate as a
    prefix that the decline had pushed them to."""
    _named_at_both_tiers(smiles, name)
    assert _emit(smiles) == name


@pytest.mark.parametrize("smiles,name", [
    # (j) alone: the two methyl groups tie under (g), the lower locant goes to R, not to S
    ("C[C@H]1C[NH2+]C[C@@H](C)C1C(=O)[O-]", "(3R,5S)-3,5-dimethylpiperidin-1-ium-4-carboxylate"),
    ("C[C@@H]1C[NH2+]C[C@H](C)C1C(=O)[O-]", "(3R,5S)-3,5-dimethylpiperidin-1-ium-4-carboxylate"),
    # (g): 'hydroxy' is cited first, so it takes the lower locant, and the descriptors follow the locants
    ("O[C@H]1C[NH2+]C[C@@H](C)C1C(=O)[O-]", "(3R,5S)-3-hydroxy-5-methylpiperidin-1-ium-4-carboxylate"),
    # (g) outranks (j): '(3R,5S)-5-hydroxy-3-methyl' would give R the lower locant but 'methyl' the lower prefix
    ("O[C@@H]1C[NH2+]C[C@H](C)C1C(=O)[O-]", "(3S,5R)-3-hydroxy-5-methylpiperidin-1-ium-4-carboxylate"),
    ("C[C@H]1C[NH2+]C[C@@H](O)C1C(=O)[O-]", "(3S,5R)-3-hydroxy-5-methylpiperidin-1-ium-4-carboxylate"),
    # the other symmetric position (2,6 around the nitrogen, the carboxylate at 4)
    ("C[C@H]1CC(C(=O)[O-])C[C@@H](C)[NH2+]1", "(2R,6S)-2,6-dimethylpiperidin-1-ium-4-carboxylate"),
    ("C[C@@H]1CC(C(=O)[O-])C[C@H](C)[NH2+]1", "(2R,6S)-2,6-dimethylpiperidin-1-ium-4-carboxylate"),
    # no stereocentre: (g) alone, 'bromo' before 'methyl'
    ("Cc1c[nH+]cc(Br)c1C(=O)[O-]", "3-bromo-5-methylpyridin-1-ium-4-carboxylate"),
    ("Cc1c[nH+]cc(O)c1C(=O)[O-]", "3-hydroxy-5-methylpyridin-1-ium-4-carboxylate"),
])
def test_a_tie_in_the_numbering_goes_to_p14_4_g_then_j(smiles, name):
    """The 'ium', the carboxylate and the set of prefix locants leave a choice of numbering (the ring is
    symmetric about the carboxylate); (g) (:3307) then (j) (:3346) decide it, as they do for the neutral
    acid ('(3R,5S)-3,5-dimethylpiperidine-4-carboxylic acid') and for '(3R,5S)-3,5-dimethylpiperidin-1-ium'.
    The emitter used to take the first direction it met (the canonical-rank tie-break), which gave
    '(3S,5R)-' and '5-hydroxy-3-methyl' labelled `pin_verified`."""
    _named_at_both_tiers(smiles, name)


@pytest.mark.parametrize("smiles", [
    "C[C@H]1C[NH2+]C[C@@H](C)C1C(=O)[O-]",
    "O[C@@H]1C[NH2+]C[C@H](C)C1C(=O)[O-]",
])
def test_the_numbering_does_not_depend_on_how_the_smiles_is_written(smiles):
    """Eight random atom orders of one species give one name."""
    jar_or_skip()
    from rdkit import Chem

    from orthonym import Orthonym
    namer = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(smiles)
    names = {namer.name_tiered(s)["name"] for s in Chem.MolToRandomSmilesVect(mol, 8, randomSeed=20261013)}
    assert len(names) == 1 and next(iter(names)) != "unknown organic compound", names


@pytest.mark.parametrize("smiles,name", [
    ("C[C@H]1C[NH2+]C[C@@H](C)[C@H]1C(=O)[O-]", "(3R,4r,5S)-3,5-dimethylpiperidin-1-ium-4-carboxylate"),
    ("C[C@H]1C[NH2+]C[C@@H](C)[C@@H]1C(=O)[O-]", "(3R,4s,5S)-3,5-dimethylpiperidin-1-ium-4-carboxylate"),
])
def test_a_pseudoasymmetric_ring_takes_r_at_the_lower_locant(smiles, name):
    """Where the fourth ring carbon is pseudoasymmetric ('r'/'s') OPSIN reads no descriptor of it, so the name
    cannot be `pin_verified` (the carve-out the neutral acid uses too): the tier is `pin_unverified` and the
    constitution is OPSIN's. What stands in for the read-back, independent of the engine: the descriptors the
    name cites are the labels of RDKit's own CIP labeller (R, S and r|s), and the name without the 'r'/'s'
    reads back to a structure that conflicts with no centre of the input. The lower locant goes to R
     (j),:3346), as in the neutral acid '(3R,4r,5S)-3,5-dimethylpiperidine-4-carboxylic acid'."""
    jar_or_skip()
    import re

    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    from orthonym import Orthonym
    from tests.support.rt_assert import _independent_parse
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_unverified", r
    mol = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(mol)
    labels = sorted(a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode"))
    assert labels == sorted(re.findall(r"\d([RSrs])", name.split(")-")[0])), (labels, name)
    reduced = re.sub(r",\d[rs]", "", name)
    parsed = _independent_parse(reduced)
    assert parsed and _stereo_aware(smiles, parsed, partial=True), (reduced, parsed)


def _neutral_acid_name(smiles):
    """The default-tier name of the neutral amino acid of a zwitterion (another producer: the neutral ring's own
    numbering, which applies (g) and (j) itself), with its ending written as the zwitterion's."""
    from rdkit import Chem

    from orthonym import Orthonym
    rw = Chem.RWMol(Chem.MolFromSmiles(smiles))
    for a in rw.GetAtoms():
        if a.GetFormalCharge():
            a.SetFormalCharge(0), a.SetNumExplicitHs(0), a.SetNoImplicit(False)
    Chem.SanitizeMol(rw)
    name = Orthonym(style="pin").name_tiered(Chem.MolToSmiles(rw))["name"]
    return name.replace("piperidine-4-carboxylic acid", "piperidin-1-ium-4-carboxylate")


@pytest.mark.parametrize("smiles", [
    "C[C@H]1C[NH2+]C[C@@H](C)C1C(=O)[O-]",
    "C[C@H]1C[NH2+]C[C@@H](C)[C@H]1C(=O)[O-]",
    "O[C@@H]1C[NH2+]C[C@H](C)C1C(=O)[O-]",
    "C[C@H]1CC(C(=O)[O-])C[C@@H](C)[NH2+]1",
    "C[C@H]1C[NH2+]C[C@@H](C)[C@@H]1C(=O)[O-]",
])
def test_the_zwitterion_is_numbered_and_cited_as_its_neutral_acid(smiles):
    """The numbering and the descriptors of '<ring>-1-ium-4-carboxylate' are those of the neutral acid, which
    another producer names (the neutral ring is numbered with (g) and (j)): '(3R,5S)-3,5-dimethyl-
    piperidine-4-carboxylic acid' and '(3R,5S)-3,5-dimethylpiperidin-1-ium-4-carboxylate'."""
    jar_or_skip()
    from orthonym import Orthonym
    name = Orthonym(style="pin").name_tiered(smiles)["name"]
    assert name == _neutral_acid_name(smiles) and name != "unknown organic compound", name


@pytest.mark.parametrize("smiles", [
    "C[C@@H](O)[C@@H]1C[NH2+]C[C@H]([C@H](C)O)C1C(=O)[O-]",
    "C[C@H](O)[C@H]1C[NH2+]C[C@@H]([C@@H](C)O)C1C(=O)[O-]",
])
def test_prefixes_that_differ_only_in_a_descriptor_are_cited_r_first(smiles):
    """'[(1R)-1-hydroxyethyl]' and '[(1S)-1-hydroxyethyl]' have one alphanumerical key,:3446 ignores the
    descriptors); and (j) ('R' precedes 'S',:22589) order them. The two drawings are one species:
    the order used to be the one in which the atoms were met, so each drawing had its own name."""
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    name = r["name"]
    assert r["tier"] == "pin_verified" and name.index("[(1R)-") < name.index("[(1S)-"), r
    assert name == "(3R,5S)-5-[(1R)-1-hydroxyethyl]-3-[(1S)-1-hydroxyethyl]piperidin-1-ium-4-carboxylate"
    assert _exact(name, smiles)


@pytest.mark.parametrize("smiles,name", [
    ("C[n+]1cc([C@H](C)O)cc([C@@H](C)O)c1", "3-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-1-methylpyridin-1-ium"),
    ("C[n+]1cc([C@H](C)Cl)cc([C@@H](C)Cl)c1", "3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1-methylpyridin-1-ium"),
    ("C[C@H](O)C1CC([C@@H](C)O)C[N+](C)(C)C1",
     "3-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-1,1-dimethylpiperidin-1-ium"),
    ("C[C@H](O)c1c([C@@H](C)O)[n+](C)cn1C",
     "4-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-1,3-dimethyl-1H-imidazol-3-ium"),
    ("C[C@H](O)c1sc[n+](C)c1[C@@H](C)O", "4-[(1R)-1-hydroxyethyl]-5-[(1S)-1-hydroxyethyl]-3-methyl-1,3-thiazol-3-ium"),
    ("C[C@H](Cl)C1=C[NH+](C)CC([C@@H](C)Cl)=C1",
     "3-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1-methyl-1,2-dihydropyridin-1-ium"),
])
def test_other_ring_cation_producers_cite_r_first_and_number_by_p14_4_j(smiles, name):
    """The plain ring-cation, azolium and hydro-ring cation producers share ``_p74_ring_substituent_prefix`` with the
    zwitterion producer. They used to cite the tied tokens in the order of the atoms ('3-[(1S)-...]-5-[(1R)-...]' or
    '5-[(1S)-...]-3-[(1R)-...]' by the way the SMILES was written) and to number a symmetric ring by the canonical
    rank. Now the tokens are cited R first,:22603) and a ring whose numbering is open takes the R prefix
    at the lower locant (j),:3346; '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)',:3399), as the
    zwitterion emitter does. Each name is ``pin_verified`` at every tier and read back exactly; the many-spelling and
    many-ring evidence is in test_ring_cation_prefix_citation.py."""
    _named_at_both_tiers(smiles, name)
