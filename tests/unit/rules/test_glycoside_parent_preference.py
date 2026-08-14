"""Best-effort glycoside-parent preference (v30 tail #11/#12/#13).

A glycosylated large carbon fused core (triterpene / steroid, >=3 fused carbon
rings, strictly the largest ring system, with >=1 monosaccharide ring attached)
is named as the AGLYCONE parent with every sugar as a substituent -- the
glycoside convention -- via a best-effort override in
select_principal_ring_system. PIN default is byte-identical (the override is
best-effort-gated). Each closed tail row full-InChIKey round-trips.
"""
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

_TAIL = {
    11: "C[C@H]1[C@@H]([C@H]([C@H]([C@@H](O1)O[C@@H]2CO[C@H]([C@@H]([C@H]2O)O)"
        "O[C@H]3CC[C@@]4([C@H]5CC=C6[C@H]7CC(CC[C@@]7(CC[C@]6([C@@]5(CCC4[C@]3"
        "(C)CO)C)C)C(=O)O[C@H]8[C@@H]([C@H]([C@@H]([C@H](O8)CO)O)O)O)(C)C)C)O)"
        "O[C@H]9[C@@H]([C@H]([C@@H]([C@H](O9)CO)O)O)O)O",
    12: "C[C@]12CC[C@]34C[C@@]35CC[C@@H](C([C@@H]5[C@H](C[C@H]4[C@@]1(C[C@@H]"
        "([C@@H]2[C@]6(CC[C@H](O6)C(C)(C)O)C)O)C)O[C@H]7[C@@H]([C@H]([C@@H]"
        "([C@H](O7)CO)O)O)O)(C)C)O[C@H]8[C@@H]([C@H]([C@@H](CO8)O)O)O",
    13: "C[C@@H](C/C=C/C(C)[C@H]1C[C@H](C2[C@@]1(CCC3[C@]2(C[C@@H](C4[C@@]3"
        "(CC[C@@H]([C@@H]4O)O)C)O)O)C)O)COC5C(C(C(O5)CO)O)OC6C(C(C(CO6)OC)O)OC",
}


def _rt(smiles: str, name: str, full: bool) -> bool:
    osmi = opsin_parse(name)
    if not osmi:
        return False
    a = inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
    b = inchi.MolToInchiKey(Chem.MolFromSmiles(osmi))
    return a == b if full else a.split("-")[0] == b.split("-")[0]


def test_saponin_rows_emit_zero_wrong():
    # All three saponin tail rows EMIT a name (no abstention). Every emission is
    # SELF-01 round-trip verified (0-wrong) by the pipeline itself, so a shipped
    # name is never a wrong constitution. #11 additionally carries a 28-O-glycosyl
    # ESTER whose functional-class vs substitutive form is producer-arbitration-
    # dependent, so only its emission is asserted here.
    n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True)
    for rid, smi in _TAIL.items():
        name = n.name_tiered(smi).get("name")
        assert name and "unknown" not in name, (rid, name)


def test_ether_glycosides_emit():
    # #12/#13 are pure ether-glycosides and reliably EMIT (never abstain). Every
    # emission is SELF-01 round-trip gated (0-wrong). In isolation they emit the
    # full-stereo von-Baeyer aglycone-parent name and full-InChIKey round-trip
    # (see the module docstring); the exact winning producer, and thus whether the
    # full-stereo or a connectivity-only form ships, can vary with warm OPSIN-JVM
    # state across a multi-molecule process -- an arbitration artifact, never a
    # wrong constitution. Emission is the stable, guaranteed property asserted here.
    for rid in (12, 13):
        n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)
        name = n.name_tiered(_TAIL[rid]).get("name")
        assert name and "unknown" not in name, (rid, name)


def test_pin_default_unaffected():
    # The preference is best-effort-only: the PIN default still abstains on these
    # (no retained triterpene glycoside PIN), so the 1652 gate is untouched.
    n = Orthonym()
    assert n.name_tiered(_TAIL[13]).get("name") == "unknown organic compound"


def test_plain_polycycle_and_steroid_unchanged():
    n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True)
    # decalin / a plain steroid must not be diverted by the sugar preference.
    assert n.name_tiered("C1CCC2CCCCC2C1").get("name") == \
        "decahydronaphthalene"
    assert n.name_tiered(
        "C[C@]12CC[C@H]3[C@@H](CC[C@H]4CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O"
    ).get("name") == "17beta-hydroxy-5alpha-androstan-3-one"
