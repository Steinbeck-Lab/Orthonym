"""The benzamide producer of the drug lane (``rules.benzene``): the amide is the suffix of a
benzene parent with its N-aryl and N-heteroaryl groups as italic-N prefixes; ring-yl ethers
are '(R)oxy' prefixes named whole.

- (the Blue Book) amides are class 11; (:18875) the parent carries the
  principal characteristic group; 'benzamide (PIN)',:32691,:32695).
- (:32774) N-substituents are prefixes with the locant N; (:32849)
  "names expressing N-substitution by a phenyl group on an amide are preferred IUPAC names";
  'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)' (:32879).
- (:7446) nesting order of enclosing marks {[({})]}.
- '2-[(pyridin-3-yl)oxy]pyrazine (PIN)',:27772): an O-linked ring is '(ring-yl)oxy';
  a ring is never a carbon-count chain ('pentyloxy' for -O-(pyridin-2-yl), main) and the
  free valence of an O-ring is read from the structure, not the ring size ('(oxan-2-yl)oxy' for
  -O-(oxan-4-yl), main).
- The benzene ring is not the parent of a ring assembly:15542;:19461;
  '(1P)-2',5'-dimethoxy-6-nitro[1,1'-biphenyl]-2-carboxylic acid (PIN)',:49805), mancude and
  saturated forms of one ring alike ('1,2,3,4,5,6-hexahydro-2,2'-bipyridine (PIN)',
  :24153,:24159; the two-ring benzene + cyclohexane assembly is the exception, 'cyclohexylbenzene
  (PIN)',:24157), nor where a linear phane name may be the PIN (2),:23829): the new
  routes decline there, and the O-ring ether branch keeps its ring-size name only below the PIN.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.rules import benzene, linear_phane_screen
from tests.support.pin_tiers import assert_not_pin_labelled, assert_pin_at_both_tiers, name_default
from tests.support.rt_assert import assert_full_rt, name_is_rt_exact
from tests.support.spelling_checks import disable_spelling_rule

pytestmark = pytest.mark.opsin_gate

NILOTINIB = "CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)C=1C=NC=CC1"
NILOTINIB_PIN = ("4-methyl-N-[3-(4-methyl-1H-imidazol-1-yl)-5-(trifluoromethyl)phenyl]-3-"
                 "{[4-(pyridin-3-yl)pyrimidin-2-yl]amino}benzamide")


def _tier_row(tier, smiles):
    if tier == "pin":
        return Orthonym(style="pin").name_tiered(smiles)
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
def test_nilotinib_is_a_benzamide_at_every_public_tier(tier):
    row = _tier_row(tier, NILOTINIB)
    assert (row["name"], row["tier"]) == (NILOTINIB_PIN, "pin_verified"), row
    assert row.get("spelling_failures") == [], row
    assert_full_rt(row["name"], NILOTINIB)


@pytest.mark.parametrize("smiles,pin", [
    ("Cc1ccc(cc1Nc1nccc(n1)-c1cccnc1)C(=O)Nc1ccccc1",
     "4-methyl-N-phenyl-3-{[4-(pyridin-3-yl)pyrimidin-2-yl]amino}benzamide"),
    ("Cc1ccc(cc1)C(=O)Nc1cc(cc(c1)C(F)(F)F)-n1cnc(C)c1",
     "4-methyl-N-[3-(4-methyl-1H-imidazol-1-yl)-5-(trifluoromethyl)phenyl]benzamide"),
    ("O=C(Nc1cccc(c1)n1cncc1)c1ccccc1", "N-[3-(1H-imidazol-1-yl)phenyl]benzamide"),
    ("O=C(Nc1ccccc1)c1cccc(Nc2ncccn2)c1", "N-phenyl-3-[(pyrimidin-2-yl)amino]benzamide"),
    ("CN(C(=O)c1ccc(C)cc1)c1ccc(cc1)-n1ccnc1",
     "N-[4-(1H-imidazol-1-yl)phenyl]-N,4-dimethylbenzamide"),
    # roflumilast
    ("O=C(Nc1c(Cl)cncc1Cl)c1ccc(OC(F)F)c(OCC2CC2)c1",
     "3-(cyclopropylmethoxy)-N-(3,5-dichloropyridin-4-yl)-4-(difluoromethoxy)benzamide"),
    # amiloride (a pyrazinecarboxamide with an N-carbamimidoyl group)
    ("N=C(N)NC(=O)c1nc(Cl)c(N)nc1N", "3,5-diamino-N-carbamimidoyl-6-chloropyrazine-2-carboxamide"),
    ("OC(=O)c1ccc(OC2CCOCC2)cc1", "4-[(oxan-4-yl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(OC2COCCN2)cc1", "4-[(morpholin-3-yl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(Oc2ccc3[nH]ccc3c2)cc1", "4-[(1H-indol-5-yl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(Oc2ncccn2)cc1", "4-[(pyrimidin-2-yl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(Sc2ccccn2)cc1", "4-[(pyridin-2-yl)sulfanyl]benzoic acid"),
    ("OCCc1ccc(Nc2ncccn2)cc1", "2-{4-[(pyrimidin-2-yl)amino]phenyl}ethan-1-ol"),
    # one benzene ring and one cyclohexane ring: substitutive:24153, 'cyclohexylbenzene
    # (PIN)':24157), so the benzamide parent is the PIN
    ("O=C(Nc1ccccn1)c1ccc(cc1)C1CCCCC1", "4-cyclohexyl-N-(pyridin-2-yl)benzamide"),
])
def test_the_drug_lane_pins(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", [
    ("O=C(Nc1ccccc1)c1ccccc1", "N-phenylbenzamide"),
    ("Cc1ccc(cc1)C(=O)Nc1ccccc1", "4-methyl-N-phenylbenzamide"),
    ("O=C(Nc1cccc(c1)C(F)(F)F)c1ccccc1", "N-[3-(trifluoromethyl)phenyl]benzamide"),
    ("O=C(Nc1ccccc1)c1cccc(Nc2ccccc2)c1", "3-anilino-N-phenylbenzamide"),
    ("CN(C(=O)c1ccccc1)c1ccccc1", "N-methyl-N-phenylbenzamide"),
    ("ONC(=O)c1ccccc1", "N-hydroxybenzamide"),
    # a stereocentre in the N-substituent: the amide producers of rules.amides cite it inside
    # the prefix,:44643)
    ("O=C(N[C@@H]1CCCNC1)c1ccccc1", "N-[(3R)-piperidin-3-yl]benzamide"),
    ("OC(=O)c1ccc(OC2CCCCO2)cc1", "4-[(oxan-2-yl)oxy]benzoic acid"),
    ("Cc1ccc(Oc2ncccn2)cc1", "2-(4-methylphenoxy)pyrimidine"),
])
def test_names_that_were_right_stay_right(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def _amide_n_substituents(smiles):
    mol = Chem.MolFromSmiles(smiles)
    out = []
    for m in mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=O)[NX3]")):
        ring = next(set(r) for r in mol.GetRingInfo().AtomRings()
                    if len(r) == 6 and any(mol.GetBondBetweenAtoms(m[0], a) for a in r))
        out.append(benzene._detect_n_substituents(mol, m, ring))
    return out


@pytest.mark.parametrize("smiles,expected", [
    (NILOTINIB, [["3-(4-methyl-1H-imidazol-1-yl)-5-(trifluoromethyl)phenyl"]]),   # main: []
    ("CN(C(=O)c1ccc(C)cc1)c1ccc(cc1)-n1ccnc1", [["4-(1H-imidazol-1-yl)phenyl", "methyl"]]),
    ("O=C(Nc1ccccc1)c1ccccc1", [["phenyl"]]),
    # left out, as on main: a stereocentre in the branch, a branch rooted at O
    ("O=C(N[C@@H]1CCCNC1)c1ccccc1", [[]]),
    ("CON(C)C(=O)c1ccccc1", [["methyl"]]),
    # left out: the acyl ring is one ring of a ring assembly (main: [])
    ("O=C(Nc1ccccn1)c1ccc(cc1)-c1ccccc1", [[]]),
])
def test_amide_n_substituents(smiles, expected):
    assert _amide_n_substituents(smiles) == expected


def _ether_group(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = next(set(r) for r in mol.GetRingInfo().AtomRings()
                if len(r) == 6 and any(mol.GetAtomWithIdx(a).GetSymbol() == "C"
                                       and any(n.GetSymbol() == "C" and n.GetDegree() == 3
                                               and not n.IsInRing()
                                               for n in mol.GetAtomWithIdx(a).GetNeighbors())
                                       for a in r))
    o = next(n.GetIdx() for a in ring for n in mol.GetAtomWithIdx(a).GetNeighbors()
             if n.GetSymbol() == "O" and n.GetIdx() not in ring and n.GetDegree() == 2)
    res = benzene._identify_oxygen_group(mol, o, ring)
    return res.get("name") if isinstance(res, dict) else res


@pytest.mark.parametrize("smiles,expected", [
    ("OC(=O)c1ccc(OC2CCOCC2)cc1", "[(oxan-4-yl)oxy]"),         # main: '(oxan-2-yl)oxy'
    ("OC(=O)c1ccc(OC2COCCN2)cc1", "[(morpholin-3-yl)oxy]"),    # main: '(oxan-2-yl)oxy'
    ("OC(=O)c1ccc(Oc2ccccn2)cc1", "[(pyridin-2-yl)oxy]"),      # main: 'pentyloxy'
    ("OC(=O)c1ccc(OCc2ccccn2)cc1", None),                      # main: 'hexyloxy'
    ("OC(=O)c1ccc(Oc2ccc3[nH]ccc3c2)cc1", "[(1H-indol-5-yl)oxy]"),
    ("OC(=O)c1ccc(OC2CCCCO2)cc1", "[(oxan-2-yl)oxy]"),
    ("OC(=O)c1ccc(OC2CCCC2)cc1", "cyclopentyloxy"),
    ("OC(=O)c1ccc(Oc2ccccc2)cc1", "phenoxy"),
    # the benzoic acid ring is one ring of a ring assembly: no ring-yl ether from these routes
    ("OC(=O)c1ccc(Oc2ncccn2)cc1-c1ccccc1", None),
    ("OC(=O)c1ccc(OC2CCOCC2)cc1-c1ccccc1", None),
])
def test_an_o_linked_ring_is_named_by_its_structure(smiles, expected):
    assert _ether_group(smiles) == expected


#: a principal group on one ring of a ring assembly: the assembly is the parent of the PIN, so
#: these benzene-parent names are never certified (each was pin_verified at every public tier
#: before the ring-assembly screen; main: abstain at 'pin', a biphenyl-based name elsewhere)
RING_ASSEMBLY_ROWS = [
    ("O=C(Nc1cccc(c1)n1ccnc1)c1ccc(cc1)-c1ccccc1",
     "N-[3-(1H-imidazol-1-yl)phenyl]-4-phenylbenzamide"),
    ("O=C(Nc1ccccn1)c1ccc(cc1)-c1ccccc1", "4-phenyl-N-(pyridin-2-yl)benzamide"),
    ("NC(=O)c1ccc(cc1-c1ccccc1)Nc1ncccn1", "2-phenyl-4-[(pyrimidin-2-yl)amino]benzamide"),
    ("OC(=O)c1ccc(Oc2ncccn2)cc1-c1ccccc1", "2-phenyl-4-[(pyrimidin-2-yl)oxy]benzoic acid"),
    ("OC(=O)c1ccc(OC2CCOCC2)cc1-c1ccccc1", "4-[(oxan-4-yl)oxy]-2-phenylbenzoic acid"),
    ("NC(=O)c1ccc(Sc2ccccn2)cc1-c1ccccc1", "2-phenyl-4-[(pyridin-2-yl)sulfanyl]benzamide"),
    # hydro ring assemblies,:24153;:24159,:17083): each was pin_verified at every
    # public tier while the screen compared rings with their bond orders (review a performance pass); the
    # PINs "2',3',4',5'-tetrahydro-N-(pyridin-2-yl)[1,1'-biphenyl]-4-carboxamide",
    # "4-{[1',2',3',4',5',6'-hexahydro[3,4'-bipyridin]-6-yl]amino}benzoic acid" and its oxy
    # analogue read back FULL
    ("O=C(Nc1ccccn1)c1ccc(cc1)C1=CCCCC1", "4-(cyclohex-1-en-1-yl)-N-(pyridin-2-yl)benzamide"),
    ("OC(=O)c1ccc(Nc2ccc(cn2)C2CCNCC2)cc1",
     "4-{[5-(piperidin-4-yl)pyridin-2-yl]amino}benzoic acid"),
    ("OC(=O)c1ccc(Oc2ccc(cn2)C2CCNCC2)cc1", "4-{[5-(piperidin-4-yl)pyridin-2-yl]oxy}benzoic acid"),
    # a biphenyl assembly with a principal group on one ring "Retained names",
    # the Blue Book-27724: '[not 1-methoxy-4-phenylbenzene; the biphenyl ring system is
    # senior to a single benzene ring]'): the benzene writer records the label (lane W2, item 26)
    ("O=C(Nc1ccccc1)c1ccc(cc1)-c1ccccc1", "N,4-diphenylbenzamide"),
]


@pytest.mark.parametrize("smiles,non_pin", RING_ASSEMBLY_ROWS)
@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
def test_a_ring_assembly_is_never_certified_as_a_benzene_parent(tier, smiles, non_pin):
    row = _tier_row(tier, smiles)
    assert not (row["name"] == non_pin and row["tier"] == "pin_verified"), row
    if tier == "best-effort":
        assert row["name"] and name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles,non_pin", [
    # the hydro ring assembly is one unit of the PIN,:24153,:24159): PINs
    # 'N-(1',2',3',4',5',6'-hexahydro[3,4'-bipyridin]-6-yl)benzamide' and
    # "1',2',3',4',5',6'-hexahydro[2,4'-bipyridin]-5-amine" (FULL)
    pytest.param("O=C(Nc1ccc(cn1)C1CCNCC1)c1ccccc1", "N-[5-(piperidin-4-yl)pyridin-2-yl]benzamide",
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "pre-existing on main, not built by this lane's routes: a hydro ring "
                     "assembly taken apart (residual R4)"))),
    pytest.param("Nc1ccc(nc1)C1CCNCC1", "6-(piperidin-4-yl)pyridin-3-amine",
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "pre-existing on main, not built by this lane's routes: a hydro ring "
                     "assembly taken apart (residual R4)"))),
    # joined at a ring nitrogen, '2H-1,2'-bipyridine (PIN)':15634): PINs
    # "3,4,5,6-tetrahydro-2H-1,2'-bipyridine" and
    # "N-(3,4,5,6-tetrahydro-2H-[1,3'-bipyridin]-6'-yl)benzamide" (FULL)
    pytest.param("c1ccc(nc1)N1CCCCC1", "2-(piperidin-1-yl)pyridine",
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "pre-existing on main, not built by this lane's routes: a hydro ring "
                     "assembly taken apart (residual R4)"))),
    pytest.param("O=C(Nc1ccc(cn1)N1CCCCC1)c1ccccc1", "N-[5-(piperidin-1-yl)pyridin-2-yl]benzamide",
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "pre-existing on main, not built by this lane's routes: a hydro ring "
                     "assembly taken apart (residual R4)"))),
])
def test_a_hydro_ring_assembly_is_not_taken_apart_in_a_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


def test_a_phane_class_molecule_gets_no_pin_through_a_ring_yl_ether(monkeypatch):
    # four ring systems and seven nodes on one chain, the acid off it: a phane name may be the
    # PIN (2),:23829), so no tier certifies the substitutive name
    smiles = "OC(=O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cn2)cc1"
    non_pin = "4-{[5-(4-phenoxyphenoxy)pyridin-2-yl]oxy}benzoic acid"
    assert linear_phane_screen.phane_may_be_pin(Chem.MolFromSmiles(smiles)) is True
    assert_not_pin_labelled(smiles, non_pin)
    # known positive: without the screen and a label check the route certifies it
    monkeypatch.setattr(linear_phane_screen, "phane_may_be_pin", lambda mol: False)
    disable_spelling_rule(monkeypatch, "P-52.2.5.1")
    row = name_default(smiles)
    assert (row["tier"], row["name"]) == ("pin_verified", non_pin), row


#: an oxan-2-yl ether on a molecule whose PIN may have another parent, a ring assembly (rows
#: 1 and 3;:15542; row 4, a pyridine bonded to the nitrogen of a piperidine,
#::15634 with:24153, PIN "4-[(oxan-2-yl)oxy]-2-(3,4,5,6-tetrahydro-2H-[1,3'-bipyridine]-
#: 6'-carboxamido)benzoic acid", FULL) or a linear phane (row 2; (2):23829): the group
#: keeps its ring-size name, recorded as a label-only non-PIN part, so the PIN tier declines and
#: the other tiers keep the name, labelled below the PIN (on main: the same names, pin_verified
#: at every public tier)
O_RING_BELOW_THE_PIN_ROWS = [
    ("OC(=O)c1ccc(OC2CCCCO2)cc1-c1ccccc1", "4-[(oxan-2-yl)oxy]-2-phenylbenzoic acid"),
    ("OC(=O)c1cc(Oc2ccc(Oc3ccccc3)cc2)ccc1OC1CCCCO1",
     "2-[(oxan-2-yl)oxy]-5-(4-phenoxyphenoxy)benzoic acid"),
    ("O=C(Nc1ccccc1)c1ccc(OC2CCCCO2)cc1-c1ccccc1", "4-[(oxan-2-yl)oxy]-N,2-diphenylbenzamide"),
    ("OC(=O)c1ccc(OC2CCCCO2)cc1NC(=O)c1ccc(cn1)N1CCCCC1",
     "4-[(oxan-2-yl)oxy]-2-[5-(piperidin-1-yl)pyridine-2-carboxamido]benzoic acid"),
]


@pytest.mark.parametrize("smiles,name", O_RING_BELOW_THE_PIN_ROWS)
@pytest.mark.parametrize("tier", ["pin", "valid", "complete", "best-effort"])
def test_an_o_ring_ether_keeps_its_name_below_the_pin(tier, smiles, name):
    row = _tier_row(tier, smiles)
    if tier == "pin":
        assert row["tier"] == "abstain", row
    else:
        assert (row["name"], row["tier"]) == (name, "systematic_verified"), row
        assert_full_rt(name, smiles)


def test_the_label_record_is_what_lowers_the_o_ring_ether(monkeypatch):
    # known positive: without the record the same name is certified, as on main
    from orthonym.metrics import provenance
    monkeypatch.setattr(provenance, "record_non_pin_label", lambda fragment: None)
    monkeypatch.setattr(provenance, "record_general_ring_prefix", lambda: None)
    smiles, name = O_RING_BELOW_THE_PIN_ROWS[0]
    row = name_default(smiles)
    assert (row["tier"], row["name"]) == ("pin_verified", name), row


def test_an_o_ring_ether_keeps_its_ring_size_name_where_a_phane_may_be_the_pin(monkeypatch):
    # oxan-2-yl-O- and oxan-4-yl-O- on the middle ring of a four-ring, seven-node chain (the acid
    # off it): the bare oxan-2-yl keeps '(oxan-2-yl)oxy' and says that the benzene parent built
    # around it may not be the PIN (the composer of the whole name records that name,
    # test_nq_drugs_ring_size_oxy_label.py); an O-ring that name does not describe is left
    # unnamed by this branch
    from orthonym.metrics import provenance
    ring = {3, 4, 5, 20, 21, 22}
    oxan2 = Chem.MolFromSmiles("OC(=O)c1cc(Oc2ccc(Oc3ccccc3)cc2)ccc1OC1CCCCO1")
    oxan4 = Chem.MolFromSmiles("OC(=O)c1cc(Oc2ccc(Oc3ccccc3)cc2)ccc1OC1CCOCC1")
    provenance.clear_provenance()
    group = benzene._identify_oxygen_group(oxan2, 23, ring)
    assert group["name"] == "(oxan-2-yl)oxy"
    assert group["parent_may_not_be_pin"] is True
    assert provenance.get_provenance()["non_pin_labels"] == ()
    provenance.clear_provenance()
    assert benzene._identify_oxygen_group(oxan4, 23, ring) is None
    # known positive: without the screen the shared namer names both groups, and the groups
    # carry no such flag
    monkeypatch.setattr(linear_phane_screen, "phane_may_be_pin", lambda mol: False)
    provenance.clear_provenance()
    for mol in (oxan2, oxan4):
        grp = benzene._identify_oxygen_group(mol, 23, ring)
        assert not grp.get("parent_may_not_be_pin"), grp
    assert benzene._identify_oxygen_group(oxan2, 23, ring)["name"] == "[(oxan-2-yl)oxy]"
    assert benzene._identify_oxygen_group(oxan4, 23, ring)["name"] == "[(oxan-4-yl)oxy]"
    assert provenance.get_provenance()["non_pin_labels"] == ()
    provenance.clear_provenance()


def test_an_o_ring_ether_of_a_phane_class_molecule_is_never_certified():
    # the oxan-4-yl analogue: main and the lane name it through the general engine
    smiles = "OC(=O)c1cc(Oc2ccc(Oc3ccccc3)cc2)ccc1OC1CCOCC1"
    assert_not_pin_labelled(smiles, "2-[(oxan-4-yl)oxy]-5-(4-phenoxyphenoxy)benzoic acid")
