"""Ring dicarboxamides cite each N-substituent with the N-locant of its own nitrogen.

 'N-Substitution', (the Blue Book): "In di- and polyamides, except for geminal
diamides, N locants with superscripted arabic numbers, which are the locants of the parent structure, are used to
differentiate the nitrogen atoms"; 'N1-methylbutanediamide (PIN)', 'N3-ethyl-N1-methylnaphthalene-1,3-disulfonamide
(PIN)'. The heterocycle producer wrote a bare 'N-' ('N-methylpyridine-2,4-dicarboxamide', pin_verified), which does
not say which nitrogen carries the group; with one primary and one N-substituted amide the ring was also numbered
without the second amide (c),:3256: every suffix group takes the lowest locants first).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("CNC(=O)c1cc(C(N)=O)ccn1", "N2-methylpyridine-2,4-dicarboxamide"),
    ("NC(=O)C1CCN(C(=O)NC)CC1", "N1-methylpiperidine-1,4-dicarboxamide"),
    ("CNC(=O)c1ccc(C(=O)NC)s1", "N2,N5-dimethylthiophene-2,5-dicarboxamide"),
    ("CCNC(=O)c1cc(C(=O)NC)ccn1", "N2-ethyl-N4-methylpyridine-2,4-dicarboxamide"),
    ("CN(C)C(=O)c1ccc(C(=O)NC)nc1", "N2,N5,N5-trimethylpyridine-2,5-dicarboxamide"),
    ("Cc1nc(C(=O)NC)cc(C(N)=O)c1", "N2,6-dimethylpyridine-2,4-dicarboxamide"),
    ("Cc1nc(C(N)=O)cc(C(=O)NC)c1", "N4,6-dimethylpyridine-2,4-dicarboxamide"),
    # benzene (rules.benzene_polyamide)
    ("CNC(=O)c1ccc(C(N)=O)cc1", "N1-methylbenzene-1,4-dicarboxamide"),
    ("CNC(=O)c1ccc(C(=O)NC)cc1", "N1,N4-dimethylbenzene-1,4-dicarboxamide"),
    ("CNC(=O)c1cccc(C(=O)NCC)c1", "N1-ethyl-N3-methylbenzene-1,3-dicarboxamide"),
    ("CNS(=O)(=O)c1cccc(S(=O)(=O)NCC)c1", "N1-ethyl-N3-methylbenzene-1,3-disulfonamide"),
    ("CNC(=O)c1ccc(Cl)c(C(N)=O)c1", "4-chloro-N1-methylbenzene-1,3-dicarboxamide"),
    ("CNC(=O)c1cc(C(=O)NC)cc(C(=O)NC)c1", "N1,N3,N5-trimethylbenzene-1,3,5-tricarboxamide"),
    # polycyclic hydrocarbons (rules.benzene_polyamide.polycyclic_polyamide_prefixes)
    ("CNC(=O)c1ccc2cc(C(N)=O)ccc2c1", "N2-methylnaphthalene-2,6-dicarboxamide"),
    ("NC(=O)c1ccc2cc(C(=O)NC)ccc2c1", "N2-methylnaphthalene-2,6-dicarboxamide"),
    ("CNC(=O)c1ccc2cc(C(=O)NC)ccc2c1", "N2,N6-dimethylnaphthalene-2,6-dicarboxamide"),
    ("CNC(=O)c1cccc2cccc(C(=O)NC)c12", "N1,N8-dimethylnaphthalene-1,8-dicarboxamide"),
    ("CCNC(=O)c1ccc2cc(C(=O)NC)ccc2c1", "N2-ethyl-N6-methylnaphthalene-2,6-dicarboxamide"),
    ("CN(C)C(=O)c1ccc2cc(C(=O)NC)ccc2c1", "N2,N2,N6-trimethylnaphthalene-2,6-dicarboxamide"),
    ("CNC(=O)c1ccc2cc(C(=O)NC)c(C)cc2c1", "N2,N6,3-trimethylnaphthalene-2,6-dicarboxamide"),
    ("CNC(=O)c1ccc2cc3cc(C(=O)NC)ccc3cc2c1", "N2,N6-dimethylanthracene-2,6-dicarboxamide"),
    ("CNC(=O)c1ccc2cc(C(N)=O)c(Cl)cc2c1", "3-chloro-N6-methylnaphthalene-2,6-dicarboxamide"),
    # fused heterocycles (rules.fused_amide_suffix.fused_n_sources)
    ("CNC(=O)c1cnc2ccc(C(=O)NC)cc2c1", "N3,N6-dimethylquinoline-3,6-dicarboxamide"),
    ("CNC(=O)c1cnc2ccc(C(N)=O)cc2c1", "N3-methylquinoline-3,6-dicarboxamide"),
    ("NC(=O)c1cnc2ccc(C(=O)NC)cc2c1", "N6-methylquinoline-3,6-dicarboxamide"),
    ("CNC(=O)c1ccc2[nH]c(C(=O)NCC)cc2c1", "N2-ethyl-N5-methyl-1H-indole-2,5-dicarboxamide"),
    ("Cn1cc(C(=O)NC)c2cc(C(=O)NC)ccc21", "N3,N5,1-trimethyl-1H-indole-3,5-dicarboxamide"),
    ("CNC(=O)c1ccc2oc3ccc(C(N)=O)cc3c2c1", "N2-methyldibenzo[b,d]furan-2,8-dicarboxamide"),
    ("CN(C)C(=O)c1cnc2ccc(C(=O)N(C)C)cc2c1", "N3,N3,N6,N6-tetramethylquinoline-3,6-dicarboxamide"),
    # (f) before the N-locants: 'N4,2-dimethylpentane-2,4-diamine (PIN)' (the Blue Book)
    ("CNC(=O)c1cc(C(N)=O)ccc1Cl", "4-chloro-N3-methylbenzene-1,3-dicarboxamide"),
    ("CNS(=O)(=O)c1cc(S(N)(=O)=O)ccc1Cl", "4-chloro-N3-methylbenzene-1,3-disulfonamide"),
    ("CNc1cc(N)ccc1Cl", "4-chloro-N3-methylbenzene-1,3-diamine"),
])
def test_ring_diamide_n_locants(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("CN(C)C(=O)c1ccc(C)nc1", "N,N,6-trimethylpyridine-3-carboxamide"),     # one amide: bare N
    ("Cc1nc(C(N)=O)cc(C(N)=O)c1", "6-methylpyridine-2,4-dicarboxamide"),
    ("NC(=O)c1ccc(C(N)=O)cc1", "benzene-1,4-dicarboxamide"),
    ("CN(C)C(=O)c1ccccc1", "N,N-dimethylbenzamide"),
    ("CNC(=O)c1ccc2ccccc2c1", "N-methylnaphthalene-2-carboxamide"),
    ("NC(=O)c1ccc2cc(C(N)=O)ccc2c1", "naphthalene-2,6-dicarboxamide"),
    # an acylamino N or a lactam on the ring is a prefix, not a second suffix group (c) vs (f),:3256)
    ("CC(=O)Nc1cc(C(N)=O)cs1", "5-acetamidothiophene-3-carboxamide"),
    ("O=C(Nc1cc(C(N)=O)cs1)c1ccccc1", "5-benzamidothiophene-3-carboxamide"),
    ("NC(=O)C1CNC(=O)C1", "5-oxopyrrolidine-3-carboxamide"),
    ("NC(=O)C1CC(=O)NC1C", "2-methyl-5-oxopyrrolidine-3-carboxamide"),
])
def test_one_amide_or_no_n_group_is_unchanged(smiles, name):
    assert tier_row(smiles, "pin")["name"] == name
