""" (2): the structures whose PIN is a linear phane name
(``validation.spelling.linear_phane``).

 (the Blue Book) "For the purpose of selecting preferred IUPAC names cyclic and
acyclic phane systems are defined as follows:" (2) (:23829) "linear phanes consist of four or more
rings or ring systems, two of which must be terminal, and together with acyclic atoms or chains
must consist of at least seven nodes (components)." (:23901) uses phane names for such
compounds "even though the compounds could also be named by substitutive or multiplicative
nomenclature"; (:24088) "Phane names are preferred IUPAC names rather than ring
assembly names when seven or more rings or ring systems are present."

The detector reads the structure only (RDKit); it never names anything. Fixtures: every linear
phane PIN the book prints (76 rows, structures rebuilt from the printed names by the lane's study
and checked against the book's own non-phane names where it prints one), the book's boundary rows,
and the class edges each rule sentence draws.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.linear_phane import (
    LinearPhaneVerdict,
    is_linear_phane_name,
    linear_phane_pin_expected,
)


def _verdict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return linear_phane_pin_expected(mol)


#: every linear phane PIN printed in the Blue Book (line, structure of the printed name)
BOOK_LINEAR_PHANE_PINS = [
    (14975, "c1cncc(CCc2ccc(CCc3ccc(Cc4ccncn4)cn3)cn2)c1"),  # L01
    (15001, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccccc4)nc3)cc2)cc1"),  # L02
    (15029, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccccc4)o3)cc2)cc1"),  # L03
    (15077, "c1ccc(Cc2ccc(Cc3cccc(Cc4ccccc4)n3)cn2)cc1"),  # L04
    (15085, "c1ccc(Cc2ccc(Cc3ccc(Cc4cccc(Cc5ccccc5)n4)o3)cn2)cc1"),  # L05
    (15107, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccc(Cc5ccccc5)nc4)nc3)cn2)cc1"),  # L06
    (15196, "c1ccc(Sc2ccc(Sc3ccc(Sc4ccccc4)cc3)cc2)cc1"),  # L07
    (15244, "c1ccc(Oc2ccc([Se]c3ccc(Sc4ccncc4)cc3)cc2)cc1"),  # L08
    (15266, "c1cc(Oc2ccc([Se]c3ccc(Sc4ccncc4)cc3)cc2)ccn1"),  # L09
    (15713, "c1c[nH]c(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6ccc(-c7ccc[nH]7)[nH]6)[nH]5)[nH]4)[nH]3)[nH]2)c1"),  # L10
    (15733, "c1ccc(-c2ccc(-c3ccc(-c4ccc(-c5ccc(-c6ccc(-c7ccccc7)cc6)c(-c6cccc(-c7ccccc7)c6)c5)cc4-c4cccc(-c5ccc(-c6ccccc6)cc5)c4)cc3)cc2)cc1"),  # L11
    (17110, "c1ccc(C2CCC(C3CCC(c4ccc(C5CCC(C6CCC(C7CCCCC7)CC6)CC5)cc4)CC3)CC2)cc1"),  # L12
    (18915, "O=C(O)c1ccc(Cc2ccc(C(c3ccc(Cc4ccncc4)cc3)c3cccc(Cc4ccc(C(=O)O)cc4)c3)cc2)cc1"),  # L13
    (19325, "C[Si](C)(C)c1ccc(Cc2ccc(Cc3ccc(CC4=CC=CCO4)cc3)cc2)cc1"),  # L14
    (20029, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccncc4)cc3)cc2)cc1"),  # L15
    (20033, "c1ccc(Cc2ccc(Cc3ccc(Cc4cc[siH]cc4)cc3)cc2)cc1"),  # L16
    (20038, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccco4)cc3)cc2)cc1"),  # L17
    (20042, "c1ccc(Cc2ccc(Cc3ccc(Cc4cccs4)cc3)cc2)cc1"),  # L18
    (20046, "c1ccc2nc(Cc3ccc(Cc4ccc(Cc5ccncc5)cc4)cc3)ccc2c1"),  # L19
    (20050, "c1cc(Cc2ccc(Cc3ccc(Cc4ccncc4)cc3)cc2)ccn1"),  # L20
    (20058, "C1=CC=C(Cc2ccc(Cc3ccc(Cc4ccncc4)cc3)cc2)NC=C1"),  # L21
    (20077, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccncc4)cc3)cc2)nc1"),  # L22
    (20088, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccncc4)cc3)cc2)[siH]c1"),  # L23
    (20093, "c1ccc2c(Cc3ccc(Cc4ccc(Cc5ccncc5)cc4)cc3)cnnc2c1"),  # L24
    (20388, "c1ccc2ncc(Cc3ccc(Cc4ccc(Cc5ccncc5)cc4)cc3)cc2c1"),  # L25
    (20393, "c1coc(Cc2ccc(Cc3ccc(Cc4ccncc4)cc3)cc2)c1"),  # L26
    (20397, "c1csc(Cc2ccc(Cc3ccc(Cc4ccncc4)cc3)cc2)c1"),  # L27
    (20405, "c1ccc(CCc2ccc(Cc3ccc(Cc4ccc(Cc5ccccc5)cc4)cc3)cc2)cc1"),  # L28
    (20445, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccc(Cc5ccccc5)cc4)cc3)cc2)cc1"),  # L29
    (20451, "c1ccc(CCc2ccc(Cc3ccc(Cc4ccc(Cc5ccncc5)cc4)cc3)nc2)cc1"),  # L30
    (20457, "c1ccc(Cc2ccc(CCc3ccc(Cc4ccc(Cc5ccncc5)cc4)cc3)nc2)cc1"),  # L31
    (20465, "c1ccc(CCCc2ccc(Cc3ccc(Cc4ccc(Cc5ccccc5)cc4)cc3)cc2)cc1"),  # L32
    (20470, "c1ccc(Cc2ccc(CCCc3ccc(Cc4ccc(Cc5ccccc5)cc4)cc3)cc2)cc1"),  # L33
    (20480, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccc(Cc5ccc(Cc6ccncc6)cc5)cc4)o3)s2)cc1"),  # L34
    (20486, "c1ccc(Cc2ccc(Cc3ccc(Cc4ccc(Cc5ccc(Cc6ccncc6)cc5)cc4)s3)o2)cc1"),  # L35
    (20517, "c1ccc(Cc2cccc(Cc3cccc(Cc4ccccc4)c3)c2)cc1"),  # L36
    (20523, "c1ccc(Cc2ccc(Cc3cccc(Cc4ccccc4)c3)cc2)cc1"),  # L37
    (20531, "c1ccc(Cc2ccc(Cc3cccc(Cc4ccncc4)c3)cc2)cc1"),  # L38
    (20537, "c1ccc(Cc2cccc(Cc3ccc(Cc4ccncc4)cc3)c2)cc1"),  # L39
    (20547, "c1ccc(Oc2ccc(Cc3ccc(Sc4ccncc4)cc3)cc2)cc1"),  # L40
    (20553, "c1ccc(Cc2ccc(Cc3ccc(Sc4ccncc4)cc3)cc2)cc1"),  # L41
    (20563, "c1cc(Oc2ccc(CCCc3ccc(Cc4ccc(Oc5ccncc5)cc4)cc3)cc2)ccn1"),  # L42
    (20569, "c1cc(Oc2ccc(Cc3ccc(CCCc4ccc(Sc5ccncc5)cc4)cc3)cc2)ccn1"),  # L43
    (20577, "c1ccc(-c2ccc(-c3ccc(Oc4cccc(-c5cccc(-c6ccccn6)n5)n4)cc3)cc2)cc1"),  # L44
    (21056, "C(#Cc1ccc(CCc2ccc(C=Cc3ccccc3)cc2)cc1)c1ccccc1"),  # L45
    (21060, "C(=Cc1ccc(CCc2ccc(CCc3ccccc3)cc2)cc1)c1ccccc1"),  # L46
    (21064, "C(#Cc1ccc(CCc2ccc(CCc3ccccc3)cc2)cc1)c1ccccc1"),  # L47
    (21102, "C(=Cc1ccc(CCc2ccc(C=Cc3ccccc3)cc2)cc1)c1ccccc1"),  # L48
    (21193, "c1ccc(Cc2ccc(SCc3ccc(Oc4ccccc4)cc3)cc2)cc1"),  # L49
    (21197, "c1ccc(Oc2ccc(CCc3ccc(Sc4ccccc4)cc3)cc2)cc1"),  # L50
    (21286, "c1ccc(Oc2ccc(CSc3ccc([Se]c4ccccc4)cc3)cc2)cc1"),  # L51
    (21290, "c1ccc(Oc2ccc(C[Se]c3ccc(Sc4ccccc4)cc3)cc2)cc1"),  # L52
    (21346, "Oc1ccc(Oc2ccc(Oc3ccc(Cc4ccccc4O)cc3)cc2)cc1"),  # L53
    (21352, "Oc1ccc(Cc2ccc(Oc3ccc(Oc4ccc(O)cc4)cc3)cc2)cc1"),  # L54
    (21409, "C(#Cc1ccc(CCc2ccccc2)cc1)Cc1ccc(C=Cc2ccc(Cc3ccccc3)cc2)cc1"),  # L55
    (21416, "C(#Cc1ccc(CCCc2ccc(C=Cc3ccc(Cc4ccccc4)cc3)cc2)cc1)c1ccccc1"),  # L56
    (21421, "C(#Cc1ccc(C=CCc2ccc(C=Cc3ccccc3Cc3ccccc3)cc2)cc1)c1ccccc1"),  # L57
    (21427, "C(#Cc1ccc(C=Cc2ccccc2)cc1)Cc1ccc(C=Cc2ccccc2Cc2ccccc2)cc1"),  # L58
    (21442, "C1=CC(Cc2ccc(Cc3ccc(Cc4ccccc4)cc3)cc2)Nc2ccccc21"),  # L59
    (21446, "C1=C(Cc2ccc(Cc3ccc(Cc4ccccc4)cc3)cc2)Nc2ccccc2C1"),  # L60
    (21631, "Clc1cc2ccccc2cc1C(c1ccc(Cc2ccc(Cc3ccccc3)cc2)cc1)c1ccc2ccccc2c1"),  # L61
    (21717, "Clc1cc(C(c2ccc(Cc3ccc(Cc4ccccc4)cc3)cc2)c2ccc3ccccc3c2Br)cc2ccccc12"),  # L62
    (22255, "Clc1c(Br)c(C(c2ccc(Cc3ccc(Cc4ccccc4)cc3)cc2)c2cc3ccccc3c(Br)c2Br)cc2ccccc12"),  # L63
    (23282, "c1ccc(Oc2cccc(Oc3cccc(Oc4ccccc4)c3)c2)cc1"),  # L64
    (23299, "c1cc(OCCOc2cocc2OCCOc2cocc2OCCOc2ccoc2)co1"),  # L65
    (23307, "c1cc(OCCOc2cocc2OCCOc2cocc2OCCOc2cocc2OCCOc2ccoc2)co1"),  # L66
    (23917, "c1ccc(-c2cccc(-c3cccc(-c4cncc(-c5cccc(-c6cccc(-c7ccccc7)c6)c5)c4)c3)c2)cc1"),  # L67
    (23959, "c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"),  # L68
    (23965, "c1cc(SCCSc2cocc2SCCSc2cocc2SCCSc2ccoc2)co1"),  # L69
    (26409, "Nc1ccc(Nc2ccc(Nc3ccc(N(c4ccc(N)cc4)c4ccc(N)cc4)cc3)cc2)cc1"),  # L70
    (27846, "c1ccc(Oc2cccc(Sc3cccc([Se]c4ccccc4)c3)c2)cc1"),  # L71
    (27898, "c1ccc(Sc2cccc(SSc3cccc(Sc4ccccc4)c3)c2)cc1"),  # L72
    (27931, "c1ccc(Sc2cccc(SSc3cccc([Te]c4ccccc4)c3)c2)cc1"),  # L73
    (31927, "O=C(Oc1cccc(C(=O)Oc2cccc(C(=O)Oc3cccc(C(=O)Oc4ccccc4)c3)c2)c1)c1ccccc1"),  # L74
    (31933, "COC(=O)c1cccc(C(=O)OCCOC(=O)c2cccc(C(=O)OCCOC(=O)c3cccc(C(=O)OCCOC(=O)c4cccc(C(=O)OC)c4)c3)c2)c1"),  # L75
    (50787, "O=C(O)C1CCC(Oc2ccc(C=COC=Cc3ccc(OC4CCC(C(=O)O)CC4OC=Cc4ccccc4)cc3)cc2)C(OC=Cc2ccccc2)C1"),  # L76
]


@pytest.mark.parametrize("line,smiles", BOOK_LINEAR_PHANE_PINS,
                         ids=[str(r[0]) for r in BOOK_LINEAR_PHANE_PINS])
def test_every_book_linear_phane_pin_is_detected(line, smiles):
    verdict = _verdict(smiles)
    assert isinstance(verdict, LinearPhaneVerdict), (line, smiles)
    assert verdict.ring_systems >= 4 and verdict.nodes >= 7


#: book PIN rows (and simple edges) that are not linear phanes; each comment gives the reason
NOT_LINEAR_PHANES = [
    ("c1ccc(-c2cncc(-c3ccccc3)c2)cc1", "23909 3,5-diphenylpyridine: 3 rings"),
    ("c1ccc(-c2cccc(-c3cncc(-c4cccc(-c5ccccc5)c4)c3)c2)cc1",
     "23913 3,5-di([1,1'-biphenyl]-3-yl)pyridine: 5 rings, 5 nodes"),
    ("c1ccc(Oc2ccccc2)cc1", "23921 1,1'-oxydibenzene: 2 rings"),
    ("c1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1", "23925 multiplicative PIN: 3 rings, 5 nodes"),
    ("c1occc1SCCSc1cocc1SCCSc1ccoc1", "23963 multiplicative PIN: 3 rings, 11 nodes"),
    ("c1ccc(C2CCC(C3CCC(c4ccc(C5CCC(C6CCCCC6)CC5)cc4)CC3)CC2)cc1",
     "17102 six rings, six nodes: 'seven nodes are required' (:17106)"),
    ("c1ccc(C2CCCCC2)cc1", "24157 cyclohexylbenzene: 2 rings"),
    ("Nc1ccc(Nc2ccc(Nc3ccccc3)cc2)cc1", "26404 3 rings"),
    ("O=C(Oc1cccc(C(=O)Oc2ccccc2)c1)c1ccccc1", "31876 phenyl 3-(benzoyloxy)benzoate: 3 rings"),
    ("Oc1ccc(SSc2ccc(O)cc2)cc1", "27892 2 rings"),
    ("C(OCC1CCCCC1)C1CCC(CC2CCC(CC3CCCCC3)CC2)CC1",
     "3547 four saturated rings on a 9-node chain: printed as a substitutive PIN"),
    ("c1ccc(-c2ccc(-c3ccc(-c4ccccc4)cc3)cc2)cc1", "p-quaterphenyl: 4 rings, 4 nodes"),
    ("c1ccc(Cc2ccc(Cc3ccc(-c4ccccc4)cc3)cc2)cc1", "4 rings, 6 nodes"),
    ("c1ccc(Oc2ccccc2)cc1.c1ccc(Oc2ccccc2)cc1", "two components of 2 rings each"),
]


@pytest.mark.parametrize("smiles,why", NOT_LINEAR_PHANES, ids=[r[1].split(":")[0] for r in NOT_LINEAR_PHANES])
def test_rows_that_are_not_linear_phanes(smiles, why):
    assert _verdict(smiles) is None, why


def test_seven_nodes_count_every_chain_atom():
    # (:14857) "a numerical term... indicating the number of nodes (including those
    # designating superatoms)";:23959 four benzenes and three O are a heptaphane
    assert _verdict("c1ccc(Oc2ccc(Oc3ccc(-c4ccccc4)cc3)cc2)cc1") is None      # 4 rings + 2 = 6
    v = _verdict("c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1")               # 4 rings + 3 = 7
    assert (v.ring_systems, v.nodes) == (4, 7)


def test_a_ring_assembly_member_is_a_node_and_a_fused_system_is_one():
    #:23917 seven rings joined by bonds: a heptaphane;:20046 'quinolina' is one amplificant
    v = _verdict("c1ccc(-c2cccc(-c3cccc(-c4cncc(-c5cccc(-c6cccc(-c7ccccc7)c6)c5)c4)c3)c2)cc1")
    assert (v.ring_systems, v.nodes) == (7, 7)
    v = _verdict("c1ccc2nc(Cc3ccc(Cc4ccc(Cc5ccncc5)cc4)cc3)ccc2c1")
    assert (v.ring_systems, v.nodes) == (4, 7)
    # the same chain with a naphthalene instead of two linked benzenes: 3 ring systems
    assert _verdict("c1ccc2cc(Cc3ccc(Cc4ccncc4)cc3)ccc2c1") is None


def test_a_spiro_system_with_a_fused_component_is_no_amplificant():
    # (a)(2) (:14910) "spiro ring systems with at least one fused ring system or
    # polycycloalkane ring system" are not allowed: the chain ends at that ring system
    assert _verdict("c1ccc(Oc2ccc(Oc3ccc(OC4CCC5(CC4)CCc4ccccc45)cc3)cc2)cc1") is None
    # a spiro system of monocycles ('spiro alkanes',:14904) is one amplificant
    v = _verdict("c1ccc(Oc2ccc(Oc3ccc(OC4CCC5(CC4)CCCCC5)cc3)cc2)cc1")
    assert (v.ring_systems, v.nodes) == (4, 7)


def test_a_chain_atom_needs_a_skeletal_replacement_prefix():
    # (:15246) lists the 'a' elements of a phane skeleton; mercury is not one
    assert _verdict("c1ccc(Oc2ccc(Oc3ccc([Hg]c4ccccc4)cc3)cc2)cc1") is None
    assert _verdict("c1ccc(Oc2ccc([Se]c3ccc(Sc4ccncc4)cc3)cc2)cc1") is not None   #:15244


def test_saturated_rings_count_when_one_ring_system_is_not_saturated():
    #:17110 '1(1),4(1,4)-dibenzena-2,3,5,6(1,4),7(1)-pentacyclohexanaheptaphane (PIN)'
    v = _verdict("c1ccc(C2CCC(C3CCC(c4ccc(C5CCC(C6CCC(C7CCCCC7)CC6)CC5)cc4)CC3)CC2)cc1")
    assert (v.ring_systems, v.nodes) == (7, 7)


def test_the_phane_must_carry_the_principal_characteristic_group():
    # (:18875) "The senior parent structure has the maximum number of substituents
    # corresponding to the principal characteristic group (suffix)", applied to phane parents at
    #:18915 ('there are two of the principal characteristic group in the PIN and only one in the
    # other names')
    acid_on_chain_ring = "OC(=O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"
    v = _verdict(acid_on_chain_ring)
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("acid", 1, 1, False)
    # a CH2-COOH arm: the acid's own parent (acetic acid) carries it, the phane cannot; the
    # chain stays whole in acetic acid's substituent, which the PIN may cite as a linear phane
    # prefix (:19325;: the verdict stands as a prefix (fail closed, ruling on F2)
    v = _verdict("OC(=O)Cc1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1")
    assert (v.ring_systems, v.nodes, v.principal_class, v.expressed, v.as_prefix) == \
        (4, 7, "acid", 0, True)
    #:18915 the path through both carboxylic acids
    v = _verdict("O=C(O)c1ccc(Cc2ccc(C(c3ccc(Cc4ccncc4)cc3)c3cccc(Cc4ccc(C(=O)O)cc4)c3)cc2)cc1")
    assert (v.principal_class, v.expressed, v.as_prefix) == ("acid", 2, False)


def test_the_phane_must_cite_as_many_as_any_other_parent():
    # (:18875) "the maximum number": every ring system, ring assembly and acyclic chain
    # of the structure is a candidate parent; a link the phane absorbs ('aza', 'oxo' + 'aza') is a
    # suffix of a ring parent (:26404 'N1-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine (PIN)'
    # cites its link N as N4)
    # pyrimidine-2,4-diamine cites the NH2 and the N4 link, the phane only the NH2; the parent
    # holds a ring of the chain, and what is left (three ring systems) is no linear phane
    assert _verdict("Nc1nc(Nc2ccc(OCc3ccc(-c4ccccc4)cc3)cc2)ccn1") is None
    # benzene-1,4-dicarboxamide cites both amides, the phane one
    assert _verdict("NC(=O)c1ccc(C(=O)Nc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1") is None
    # a ring assembly is one parent,:15550;:26332 '[1,1'-biphenyl]-4,4'-diamine (PIN)')
    assert _verdict("Nc1ccc(-c2ccc(Nc3ccc(Oc4ccc(Oc5ccccc5)cc4)cc3)cc2)cc1") is None
    # the assembly cites 2 amines against the phane's 1 and holds a WHOLE ring system of the
    # chain (not just its link atom), so _node_atoms must read the ring system's full atom set
    # to see the assembly holds a node of the only chain: biphenyl-3,4'-diamine
    assert _verdict("Nc1ccc(-c2cc(Oc3ccccc3)c(N)cc2Oc2ccc(Oc3ccccc3)cc2)cc1") is None
    # an acyclic chain is one parent: butanedioic acid cites two acids, the phane one; the
    # chain stays whole in its substituent, a linear phane prefix of the PIN (:19325;
    #: the verdict stands as a prefix (fail closed, ruling on F2)
    v = _verdict("OC(=O)CC(C(=O)O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccc(C(=O)O)cc4)cc3)cc2)cc1")
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("acid", 1, 3, True)
    # a tie goes to the phane::26409 two amines in the phane and in the substitutive
    # benzene-1,4-diamine;:31927 one ester in each
    v = _verdict("Nc1ccc(Nc2ccc(Nc3ccc(N(c4ccc(N)cc4)c4ccc(N)cc4)cc3)cc2)cc1")
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("amine", 2, 6, False)
    v = _verdict("O=C(Oc1cccc(C(=O)Oc2cccc(C(=O)Oc3cccc(C(=O)Oc4ccccc4)c3)c2)c1)c1ccccc1")
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("ester", 1, 4, False)


#: review a performance pass (finding F2) and the pendant chalcogen ketones of finding F1: another parent
#: cites more of the principal characteristic group than the phane,:18875), but
#: the qualifying chain (four benzenes, three O) stays whole in that parent's substituent
PARENT_OFF_THE_CHAIN = [
    ("OC(=O)Cc1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "acid", 0, 1),         # acetic acid
    ("COC(=O)Nc1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "ester", 0, 1),       # carbamic acid
    ("NC(=O)Nc1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "amide", 0, 1),        # urea
    ("CC(=O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "ketone", 0, 1),        # ethan-1-one
    ("CC(=S)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "ketone", 0, 1),        # ethane-1-thione
    ("CC(=[Se])c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "ketone", 0, 1),     # ethane-1-selone
    # pyrimidine-2,4-diamine, a ring off the four-ring chain of the N4 group
    ("Nc1nc(Nc2ccc(Oc3ccc(Oc4ccc(Oc5ccccc5)cc4)cc3)cc2)ccn1", "amine", 1, 2),
    # butanedioic acid
    ("OC(=O)CC(C(=O)O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccc(C(=O)O)cc4)cc3)cc2)cc1", "acid", 1, 3),
]


@pytest.mark.parametrize("smiles,cls,expressed,occurrences", PARENT_OFF_THE_CHAIN)
def test_a_chain_whole_outside_the_winning_parent_fails_closed(smiles, cls, expressed, occurrences):
    #:19325 'trimethyl[1^2H-1(6)-pyrana-3,5(1,4),7(1)-tribenzenaheptaphan-7^4-yl]silane (PIN)
    # (Si is senior to O)': the one printed PIN whose parent is not the phane cites the chain as
    # a linear phane prefix; (:16148) "Substituent prefixes derived from phane
    # systems". No printed row decides it for a parent that wins by, so the
    # substitutive name is not certified: the verdict stands, as a prefix (controller ruling of
    # 2026-10-05 on the final review, finding F2: fail closed)
    v = _verdict(smiles)
    assert v is not None, smiles
    assert (v.ring_systems, v.nodes, v.principal_class, v.expressed, v.occurrences, v.as_prefix) \
        == (4, 7, cls, expressed, occurrences, True)


def test_any_tied_parent_that_leaves_the_chain_whole_fails_closed():
    # four OH: an ethane-1,2-diol arm on a terminal ring and a propane-1,3-diol whose C2 is a
    # chain node each cite two leaves them tied; the criteria after it are not read
    # here); the arm's parent leaves the chain whole, so the PIN may cite it as a prefix
    v = _verdict("OCC(O)c1ccc(Cc2ccc(C(CO)(CO)c3ccc(Cc4ccccc4)cc3)cc2)cc1")
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("hydroxy", 0, 4, True)
    # the propane-1,3-diol alone: its C2 splits the only chain into two-ring pieces
    assert _verdict("c1ccc(Cc2ccc(C(CO)(CO)c3ccc(Cc4ccccc4)cc3)cc2)cc1") is None


def test_a_winning_parent_holds_the_nitrogen_of_the_amine_it_cites():
    # the N-methyl amine N is a chain node; every parent citing it (a ring beside it, or
    # methanamine) holds that N, so no qualifying chain stays whole
    assert _verdict("c1ccc(Cc2ccc(N(C)c3ccc(Cc4ccccc4)cc3)cc2)cc1") is None


def test_a_carbonyl_with_two_heteroatom_neighbours_is_no_pseudoketone():
    # a urea with an acyclic nitrogen is "an amide of carbonic acid",:33320;
    # Table 4.1 class 11,:18184): 'piperidine-1-carboxamide (PIN)' (:32685), not a pseudoketone
    #,:29312: no acyclic N on the carbonyl of a pseudoketone; (b),:28263, an
    # acyl group on a ring N). Its C=O is a chain node here, so the phane cites no amide.
    spellings = ("O=C(Nc1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1)N1CCC(c2ccccc2)CC1",
                 "c1ccc(C2CCN(C(=O)Nc3ccc(Oc4ccc(Oc5ccccc5)cc4)cc3)CC2)cc1")
    mols = [Chem.MolFromSmiles(s) for s in spellings]
    assert Chem.MolToInchiKey(mols[0]) == Chem.MolToInchiKey(mols[1])
    assert [linear_phane_pin_expected(m) for m in mols] == [None, None]
    # a diaryl carbonate on the chain: an ester (class 9,:18182), 'oxo' + 'dioxa' in the phane
    assert _verdict("c1ccc(Oc2ccc(OC(=O)Oc3ccc(Oc4ccccc4)cc3)cc2)cc1") is None
    # a urea on a skeletal ring N, its C=O off the chain: the phane's carboxamide suffix (a tie)
    v = _verdict("NC(=O)N1CCC(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)CC1")
    assert (v.principal_class, v.expressed, v.occurrences) == ("amide", 1, 1)


def test_a_carbonyl_between_two_ring_nitrogens_is_a_pseudoketone():
    # no acyclic nitrogen is left for an amide: (:29312) pseudoketones have a carbonyl
    # joined to two heteroatoms none of which is an acyclic N; (:29370),
    # (:33125) hidden amides;:29544 'di(1H-imidazol-1-yl)methanethione (PIN)'. The C=O is a
    # skeletal '-one' of the phane, as of the methanone parent: a tie, the phane
    spellings = ("O=C(N1CCC(c2ccc(Oc3ccccc3)cc2)CC1)N1CCC(c2ccccc2)CC1",
                 "c1ccc(C2CCN(C(=O)N3CCC(c4ccc(Oc5ccccc5)cc4)CC3)CC2)cc1")
    mols = [Chem.MolFromSmiles(s) for s in spellings]
    assert Chem.MolToInchiKey(mols[0]) == Chem.MolToInchiKey(mols[1])
    for mol in mols:
        v = linear_phane_pin_expected(mol)
        assert (v.ring_systems, v.nodes) == (5, 7)
        assert (v.principal_class, v.expressed, v.occurrences) == ("ketone", 1, 1)
    # a piperazine on one side (its other N a ring atom too), and the thiourea (:29581
    # '1,1'-carbonothioyldi(pyridin-2(1H)-one) (PIN)', a C=S between two ring N in
    for smiles in ("O=C(N1CCC(c2ccc(Oc3ccccc3)cc2)CC1)N1CCN(c2ccccc2)CC1",
                   "S=C(N1CCC(c2ccc(Oc3ccccc3)cc2)CC1)N1CCC(c2ccccc2)CC1"):
        v = _verdict(smiles)
        assert (v.principal_class, v.expressed, v.occurrences) == ("ketone", 1, 1)
    # a sulfonyl between two ring nitrogens: no acyclic nitrogen for a sulfonamide (Table 4.1
    # class 11), so no suffix class, as for a sulfonyl on one ring nitrogen
    v = _verdict("O=S(=O)(N1CCC(c2ccc(Oc3ccccc3)cc2)CC1)N1CCC(c2ccccc2)CC1")
    assert v is not None and (v.principal_class, v.expressed) == (None, 0)


def test_a_c_nh_on_ring_nitrogens_is_an_imine():
    # Table 4.1 class 16 (:18189) is a carbonyl in every alternative (ketones -C-CO-C-,
    # pseudoketones -C-CO-X, X-CO-X, -CO-X-CO-); class 20 (:18193) "Imines, R=NH or R=N-R'".
    # A guanidine between two ring nitrogens: the phane cites its carbon as '-imine'
    v = _verdict("N=C(N1CCC(c2ccc(Oc3ccccc3)cc2)CC1)N1CCC(c2ccccc2)CC1")
    assert (v.principal_class, v.expressed, v.occurrences) == ("imine", 1, 1)
    # with an acetyl on the terminal benzene the ketone (class 16) is the principal
    # characteristic group: its carbon is no skeletal atom of the phane, the ethan-1-one parent
    # cites it,:18875), and the C=NH is a prefix; the chain stays whole in the
    # ethanone's substituent, a linear phane prefix of the PIN (:19325;; ruling on F2)
    v = _verdict("CC(=O)c1ccc(C2CCN(C(=N)N3CCC(c4ccc(Oc5ccccc5)cc4)CC3)CC2)cc1")
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("ketone", 0, 1, True)
    # the same for an amidine on one ring nitrogen whose carbon is a chain node
    v = _verdict("CC(=O)c1ccc(C2CCN(C(=N)Cc3ccc(Oc4ccccc4)cc3)CC2)cc1")
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("ketone", 0, 1, True)


def test_an_amide_linking_two_chain_rings_keeps_the_substitutive_parent():
    # reading R2 (controller decision 1): the amide's C and N are both chain nodes, so a phane
    # name could cite it only as 'oxo' + 'aza' -- an acyclic C=O on a skeletal N is no
    # pseudoketone (:1878 "except for nitrogen") -- and picks the benzamide
    nilotinib = "CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)C=1C=NC=CC1"
    imatinib = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
    assert _verdict(nilotinib) is None
    assert _verdict(imatinib) is None


def test_ketones_and_pseudoketones_on_the_skeleton_are_cited_by_the_phane():
    # venadaparib: the phthalazinone C=O (a cyclic pseudoketone) and the C=O on the azetidine N
    # (:1878 "a carbonyl group linked to a heteroatom belonging to a ring") are both on the
    # skeleton: '...cyclopropanaoctaphane-1^4,4(1^3H)-dione'
    venadaparib = "C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O"
    v = _verdict(venadaparib)
    assert (v.ring_systems, v.nodes, v.principal_class, v.expressed) == (4, 8, "ketone", 2)
    olaparib = "O=C(c1cc(Cc2n[nH]c(=O)c3ccccc23)ccc1F)N1CCN(C(=O)C2CC2)CC1"
    v = _verdict(olaparib)
    assert (v.ring_systems, v.nodes, v.principal_class, v.expressed) == (4, 7, "ketone", 3)


def test_an_amine_on_a_cyclic_amidine_is_an_amine():
    # Table 4.1 (:18162): a ring C=N belongs to the heterocycle (class 21,:18197); the
    # exocyclic NH of a 4,5-dihydro-1,3-oxazol-2-yl group is an amine (class 19,:18192), not
    # an amidine (class 11).
    # Tucatinib: its two NH links are chain nodes, so the phane cites neither amine and the
    # quinazoline-4,6-diamine parent wins
    tucatinib = "Cc1cc(Nc2ncnc3ccc(NC4=NC(C)(C)CO4)cc23)ccc1Oc1ccn2ncnc2c1"
    assert _verdict(tucatinib) is None


def test_a_cyclic_anhydride_is_a_heterocyclic_dione():
    # Table 4.1 class 8 (:18181): "substitutive nomenclature is used for cyclic anhydrides that
    # are named as heterocycles (see 16 below)": its two C=O are cited as '-dione' (class 16,
    #:18189)
    v = _verdict("O=C1OC(=O)c2cc(Oc3ccc(Oc4ccc(Oc5ccccc5)cc4)cc3)ccc21")
    assert (v.principal_class, v.expressed, v.occurrences) == ("ketone", 2, 2)


#: the:23959 heptaphane with one group on a terminal ring ('{}' is the group's SMILES head)
ON_A_CHAIN_RING = "{}c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"


@pytest.mark.parametrize("group", ["OC(=O)", "SC(=O)", "OC(=S)", "SC(=S)", "OOC(=O)", "N=C(O)"])
def test_chalcogen_peroxy_and_imidic_analogues_of_an_acid_are_acids(group):
    # Table 4.1 7a (:18172) "'Suffix' acids in the order carboxylic,... each followed in turn by
    # the corresponding peroxy, imidic, and hydrazonic acids. Chalcogen analogues follow each of
    # the corresponding oxygen acids"; Table 4.4 (:18603 carboperoxoic,:18615 "Carboxylic acids
    # modified by replacement with S, Se, and/or Te",:18625 carboximidic acids). On a chain ring
    # the phane cites the acid as the carboxylic acid (a tie with the benzene,
    v = _verdict(ON_A_CHAIN_RING.format(group))
    assert (v.principal_class, v.expressed, v.occurrences) == ("acid", 1, 1)


@pytest.mark.parametrize("group", ["CC(=O)", "CC(=S)", "CC(=[Se])"])
def test_a_pendant_thione_or_selone_is_a_ketone_its_own_chain_cites(group):
    # (:29504) "Chalcogen analogues of ketones, pseudoketones and heterones are named by
    # using the following suffixes" (Table 4.1 class 16); the carbon is no skeletal atom of the
    # phane, so the ethanone, ethanethione or ethaneselone parent cites it,:18875), as
    # for the C=O (decision 1, R3.2); the chain stays whole in that parent's substituent, a
    # linear phane prefix of the PIN (:19325;: a prefix verdict (ruling on F2)
    v = _verdict(ON_A_CHAIN_RING.format(group))
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("ketone", 0, 1, True)


def test_chalcogen_analogues_of_aldehydes_and_ketones_keep_their_class():
    # Table 4.1 class 15 (:18188) "Aldehydes and chalcogen analogues": a thioaldehyde on a chain
    # ring is the phane's '-carbothialdehyde'
    v = _verdict("S=C" + ON_A_CHAIN_RING.format(""))
    assert (v.principal_class, v.expressed, v.occurrences) == ("aldehyde", 1, 1)
    #:29544 'di(1H-imidazol-1-yl)methanethione (PIN)': a C=S between two ring nitrogens is a
    # pseudoketone's thione, cited by the phane on its skeleton
    v = _verdict("S=C(n1ccnc1)n1cc(-c2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)nc1")
    assert (v.principal_class, v.expressed, v.occurrences) == ("ketone", 1, 1)


@pytest.mark.parametrize("group", ["CCOC(=O)", "CCOC(=S)", "CSC(=S)"])
def test_a_thiono_or_dithio_ester_is_an_ester(group):
    # Table 4.1 class 9 (:18182) esters, before class 16: the ester claims its C=S, which is no
    # thione; the acid component on a chain ring is the phane's carboxylate (a tie)
    v = _verdict(ON_A_CHAIN_RING.format(group))
    assert (v.principal_class, v.expressed, v.occurrences) == ("ester", 1, 1)


def test_only_the_most_senior_suffix_is_the_principal_characteristic_group():
    # (:18875) counts "the principal characteristic group (suffix)", one suffix of the
    # order of Table 4.4 (:18597)
    # (:29561) "The order of seniority of ketonic suffixes is C=O > C=S > C=Se > C=Te":
    # a C=O and a C=S on the skeleton, the phane cites one '-one'
    v = _verdict("O=C(c1ccc(Oc2ccccc2)cc1)c1ccc(C(=S)c2ccc(Oc3ccccc3)cc2)cc1")
    assert (v.principal_class, v.expressed, v.occurrences) == ("ketone", 1, 1)
    # Table 4.4 entry 49 (:18836, '-SH thiol':18838): -ol before -thiol
    v = _verdict("Oc1ccc(Oc2ccc(Oc3ccc(Oc4ccc(S)cc4)cc3)cc2)cc1")
    assert (v.principal_class, v.expressed, v.occurrences) == ("hydroxy", 1, 1)
    # (:33386) "amides from carboxylic acids, including formamide, are senior to
    # urea": the urea on an acyclic N is no principal group beside the carboxamide
    v = _verdict("NC(=O)Nc1ccc(Oc2ccc(Oc3ccc(Oc4ccc(C(N)=O)cc4)cc3)cc2)cc1")
    assert (v.principal_class, v.expressed, v.occurrences) == ("amide", 1, 1)
    # Table 4.1 class 11 (:18184) "Amides [in the order of the corresponding acids...]": the
    # benzamide whose amide links two chain rings cites the senior carboxamide, the phane only the
    # two junior sulfonamides (Table 4.4 entry 16:18759 before 19:18769)
    assert _verdict("NS(=O)(=O)c1ccc(Oc2ccc(C(=O)Nc3ccc(Oc4ccc(S(N)(=O)=O)cc4)cc3)cc2)cc1") is None


#::23959's heptaphane between two amides on its terminal rings and a benzamide link
BESIDE_A_BENZAMIDE_LINK = "{0}c1ccc(Oc2ccc(C(=O)Nc3ccc(Oc4ccc({1})cc4)cc3)cc2)cc1"


@pytest.mark.parametrize("head,tail", [
    ("NC(=O)", "C(N)=O"),                                    # carboxamides (control)
    ("O=NN(C)C(=O)", "C(=O)N(C)N=O"),                        # N-methyl-N-nitroso amides
    ("O=[N+]([O-])N(C)C(=O)", "C(=O)N(C)[N+](=O)[O-]"),      # N-methyl-N-nitro amides
    ("O=C(Nn5cccc5)", "C(=O)Nn5cccc5"),                      # N-(1H-pyrrol-1-yl) amides
    ("O=C(NN5CCCCC5)", "C(=O)NN5CCCCC5"),                    # N-(piperidin-1-yl) amides
    ("NNNNC(=O)", "C(=O)NNNN"),                              # N-(triazan-1-yl) amides
])
def test_an_amide_whose_n_bears_no_hydrazine_nitrogen_is_an_amide(head, tail):
    # Table 4.1 class 11 amides (:18184) before class 12 hydrazides (:18185). A nitroso or nitro
    # nitrogen is the central atom of nitrous or nitric acid (:35880 "amides and hydrazides of
    # nitric and nitrous acids are now systematically based on nitric or nitrous amide"), so
    # the group is an amide with that prefix, not a hydrazide; a ring nitrogen cannot be part of
    # the suffix group; a longer nitrogen chain is a polyazane substituent,:24906
    # 'N-(triazan-1-yl)benzamide (PIN)'). Three amides: the phane cites the two on its terminal
    # rings, the benzamide parent one (its link is 'oxo' + 'aza' in the phane): a phane
    v = _verdict(BESIDE_A_BENZAMIDE_LINK.format(head, tail))
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == ("amide", 2, 3, False)


@pytest.mark.parametrize("head,tail", [
    ("NNC(=O)", "C(=O)NN"),                                  # hydrazides
    ("CNNC(=O)", "C(=O)NNC"),                                # N'-methyl hydrazides
    ("ONNC(=O)", "C(=O)NNO"),                                # N'-hydroxy hydrazides
])
def test_a_hydrazide_beside_a_benzamide_link_is_not_the_principal_group(head, tail):
    # class 12 (:18185): the senior amide is the benzamide link, which the phane cites only as
    # 'oxo' + 'aza'; the benzamide parent holds a chain ring, so no linear phane is left. An
    # N'-hydroxy group keeps the hydrazide, as N-hydroxy keeps the amide (:17572
    # 'N-hydroxypropanamide (PIN)')
    assert _verdict(BESIDE_A_BENZAMIDE_LINK.format(head, tail)) is None


@pytest.mark.parametrize("smiles,expected", [
    # Table 4.4 50 (:18841): -OSH:18843 before -SOH:18844. The senior one on an arm: its own
    # chain cites it (a prefix verdict); on the chain ring: the phane cites it (a tie)
    ("SOCc1ccc(Oc2ccc(Oc3ccc(Oc4ccc(SO)cc4)cc3)cc2)cc1", ("hydroperoxide", 0, 1, True)),
    ("OSCc1ccc(Oc2ccc(Oc3ccc(Oc4ccc(OS)cc4)cc3)cc2)cc1", ("hydroperoxide", 1, 1, False)),
    # Table 4.3 (c) (:18493 "oxygen atoms, then S, Se, and Te atoms, in -(O)OH and -OH
    # groups"): '-CO-SOH carbo(thioperoxoic) SO-acid':18508 before '-CO-OSH':18509
    ("OSC(=O)Cc1ccc(Oc2ccc(Oc3ccc(Oc4ccc(C(=O)OS)cc4)cc3)cc2)cc1", ("acid", 0, 1, True)),
    ("SOC(=O)Cc1ccc(Oc2ccc(Oc3ccc(Oc4ccc(C(=O)SO)cc4)cc3)cc2)cc1", ("acid", 1, 1, False)),
])
def test_the_two_spellings_of_a_thioperoxy_pair_are_not_tied(smiles, expected):
    v = _verdict(smiles)
    assert (v.principal_class, v.expressed, v.occurrences, v.as_prefix) == expected


def test_a_hydrazide_is_junior_to_an_amide():
    # Table 4.1 class 12 (:18185) "Hydrazides (in the order of the corresponding acids)", after
    # class 11; Table 4.4 entry 31 (:18791) carbohydrazides. Alone on a chain ring it is the
    # phane's '-carbohydrazide'; beside the benzamide whose amide links two chain rings it is not
    # the principal characteristic group
    v = _verdict(ON_A_CHAIN_RING.format("NNC(=O)"))
    assert (v.principal_class, v.expressed, v.occurrences) == ("hydrazide", 1, 1)
    assert _verdict("NNC(=O)c1ccc(Oc2ccc(C(=O)Nc3ccc(Oc4ccc(C(=O)NN)cc4)cc3)cc2)cc1") is None


def test_a_charged_component_fails_closed():
    # an uncompensated charge (a cation or anion class, classes 1-6) is not modelled: the
    # structure test alone decides, so the substitutive name is never certified
    v = _verdict("C[n+]1ccc(Cc2ccc(Cc3ccc(Cc4ccccc4)cc3)cc2)cc1.[I-]")
    assert v is not None and v.principal_class == "ion"
    # a charge-separated neutral group (nitro, N-oxide, azide: each charged atom has a neighbour of
    # the opposite sign) is no ion
    for smiles in ("O=[N+]([O-])c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1",
                   "[O-][n+]1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1",
                   "[N-]=[N+]=Nc1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"):
        v = _verdict(smiles)
        assert v is not None and v.principal_class is None, smiles
    # an ammonium cation is one
    v = _verdict("[NH3+]c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1")
    assert v is not None and v.principal_class == "ion"


def test_each_component_is_read_alone():
    v = _verdict("c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1.Cl")
    assert (v.ring_systems, v.nodes) == (4, 7)


@pytest.mark.parametrize("smiles", [
    "C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O",
    "O=C(O)c1ccc(Cc2ccc(C(c3ccc(Cc4ccncc4)cc3)c3cccc(Cc4ccc(C(=O)O)cc4)c3)cc2)cc1",
    "CC1=C(C=C(C(=O)NC2=CC(=CC(=C2)C(F)(F)F)N2C=NC(=C2)C)C=C1)NC1=NC=CC(=N1)C=1C=NC=CC1",
])
def test_the_atom_order_does_not_change_the_verdict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    first = linear_phane_pin_expected(mol)
    n = mol.GetNumAtoms()
    for shift in (1, 7, n - 1):
        order = [(i + shift) % n for i in range(n)][::-1]
        assert linear_phane_pin_expected(Chem.RenumberAtoms(mol, order)) == first


@pytest.mark.parametrize("smiles", [
    "O=C(Nc1ccc(Oc2ccc(Oc3ccccc3)cc2)cc1)N1CCC(c2ccccc2)CC1",           # a urea on a ring N
    "NC(=O)N1CCC(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)CC1",                   # a urea, skeletal ring N
    "O=C(N1CCC(c2ccc(Oc3ccccc3)cc2)CC1)N1CCC(c2ccccc2)CC1",              # a urea between ring N
    "O=C(NC(=O)c1ccccc1)c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1",      # a diacylamine
    "C1(CC1)NCC1CN(C1)C(=O)C=1C=C(C=CC1F)CC1=NNC(C2=CC=CC=C12)=O",       # venadaparib
    "Nc1nc(Nc2ccc(OCc3ccc(-c4ccccc4)cc3)cc2)ccn1",
])
def test_the_smiles_spelling_does_not_change_the_verdict(smiles):
    # a renumbering keeps each atom's neighbour order; a new SMILES spelling does not, and a
    # reading that depends on which neighbour a match lists first would differ between them
    mol = Chem.MolFromSmiles(smiles)
    first = linear_phane_pin_expected(mol)
    for spelling in Chem.MolToRandomSmilesVect(mol, 20, randomSeed=7):
        other = Chem.MolFromSmiles(spelling)
        assert Chem.MolToInchiKey(other) == Chem.MolToInchiKey(mol)
        assert linear_phane_pin_expected(other) == first, spelling


@pytest.mark.parametrize("name,expected", [
    ("2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphane", True),
    ("2,5,7,10,12,15-hexathia-1,16(3),6,11(3,4)-tetrafuranahexadecaphane", True),
    ("2,5,7,10,12,15,17,20-octaoxa-1,21(3),6,11,16(3,4)-pentafuranahenicosaphane", True),
    ("1,10(1),4,7(1,4)-tetrabenzenadecaphan-2-en-8-yne", True),
    ("trimethyl[1^2H-1(6)-pyrana-3,5(1,4),7(1)-tribenzenaheptaphan-7^4-yl]silane", True),
    ("1,4(1,4)-dibenzenacyclohexaphane", False),
    ("1(1,3)-benzenacycloheptadecaphane", False),
    ("triphenylphosphane", False),
    ("1-phenoxy-4-(4-phenoxyphenoxy)benzene", False),
    ("L-tryptophan", False),
])
def test_is_linear_phane_name(name, expected):
    assert is_linear_phane_name(name) is expected
