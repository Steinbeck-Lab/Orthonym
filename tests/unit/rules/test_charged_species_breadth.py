from rdkit import Chem
from rdkit.Chem import inchi
from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


def _be():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _full_rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


def test_complex_quaternary_ammonium_names():
    # branched acyclic parent bearing a ring substituent + a charged N.
    # neutral analog already names; the charge must not force an abstention.
    smi = "CCC(Cc1ccco1)C[N+](C)(C)C"
    name = _be().name_tiered(smi).get("name")
    assert name and "unknown" not in name, name
    assert _full_rt(smi, name), name


def test_simple_charged_still_name_unchanged():
    n = _be()
    assert n.name_tiered("CCCC[N+](C)(C)C").get("name") == "N,N,N-trimethylbutan-1-aminium"
    assert n.name_tiered("CCC[NH3+]").get("name") == "propan-1-aminium"
    assert n.name_tiered("CCCC(=O)[O-]").get("name") == "butanoate"
    assert n.name_tiered("C1CC[NH2+]CC1").get("name") == "piperidin-1-ium"


def test_multifragment_salt_assembles():
    # once the cation names, name_adduct assembles the two-word salt.
    smi = "CCC(Cc1ccco1)C[N+](C)(C)C.[I-]"
    name = _be().name_tiered(smi).get("name")
    assert name and "unknown" not in name and "iodide" in name, name
    assert _full_rt(smi, name), name


def test_simple_salt_unchanged():
    n = _be()
    assert n.name_tiered("C[N+](C)(C)C.[I-]").get("name") == \
        "N,N,N-trimethylmethanaminium iodide"


def test_inorganic_salt_fails_closed():
    # out of scope -> must abstain, never a guessed/partial name.:
    # name_salt used to silently drop the unnameable chlorosilanolate anion
    # and emit the partial 'beryllium strontium'. With _be
    # (general_fallback=True), the correct abstain value is None, not a
    # literal sentinel string: namer.py's documented " Composer1 Task 5
    # best-effort clean-abstain contract" (name_tiered, ~:2939-2949)
    # deliberately nulls out ANY failure name under general_fallback to avoid
    # leaking a descriptive fallback as if it were a real name (the contributor guide
    # a project rule, the sentinel-leak defect class). Verified this molecule's
    # raw fallback text is actually 'beryllium compound (not supported)'
    # (errors.py::classify_failure_limit's UNSUPPORTED_ELEMENT branch), not
    # the generic 'unknown organic compound' literal this assertion
    # originally guessed -- that string was unreachable for this SMILES
    # under any configuration (confirmed by disabling the null-out and
    # re-running: result is 'beryllium compound (not supported)', never
    # 'unknown organic compound').
    #
    # Breadth Job 2 (the user's decision, 2026-09-28): the best-effort tier now names
    # this drawing as a adduct of its three ions -- a COMPLETE name, which a
    # fresh OPSIN call reads back to exactly the drawn structure; the hazard
    # (a partial name that drops the anion) stays closed, and the label is below PIN
    #, the Blue Book, no PIN for a mixed adduct).
    from tests.support.rt_assert import _independent_parse
    smiles = "[Be+2].[O-][Si](=O)Cl.[Sr+2]"
    row = _be().name_tiered(smiles)
    name = row.get("name")
    assert name == ("1-chloro-1-oxido-2-oxa-1-silaeth-1-ene\u2014beryllium(2+)"
                    "\u2014strontium(2+) (1/1/1)"), name
    parsed = _independent_parse(name)
    assert parsed and Chem.MolToSmiles(Chem.MolFromSmiles(parsed)) == \
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles)), (name, parsed)
    assert row["is_pin"] is False and row["tier"] == "systematic_verified"
