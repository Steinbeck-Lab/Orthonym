""" a phase — general-fallback STEREO-OMISSION reclaim (0-wrong).

A best-effort (general-fallback tier) candidate whose CONSTITUTION round-trips but
whose flat name dropped the input's stereo used to be suppressed
(`full_rt_unconfirmed_general_fallback`). The gate now composes the input stereo
back on with 's RT-gated re-anchor (OPSIN's own numbering), shipping it ONLY when
the composed name recomputes to the input's full InChIKey — the identical 0-wrong bar
as every other RT-verified emission. See ``namer._try_compose_input_stereo`` and the
hook at the ``full_rt_unconfirmed_general_fallback`` suppression site.

Measured on the frozen abstain sample (`benchmarks/naming_scale/v43_abstain_sample.jsonl`):
1435 GATE_SUPPRESSED rows -> 31 stereo-only -> 7 composed, **0 wrong**; 6 reclaim
end-to-end through the engine. PIN tier (``general_fallback_tier`` False) never reaches
this branch, so the PIN gate is byte-identical (kill switch:
``ORTHONYM_STEREO_OMISSION_RECLAIM=0`` / ``namer._STEREO_OMISSION_RECLAIM``).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import namer as _namer
from orthonym.namer import Orthonym, _validity_gate_name_to_smiles
from orthonym.jvm_budget import jvm_slots
from orthonym.errors import is_failure_name


def _best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def _full_ik(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m) if m else None


# (input SMILES, exact composed name) — each verified end-to-end at 0-wrong.
RECLAIM_CASES = [
    # Suite fix j6 (TRIAGE g7 C15): the salt abstained -- the ring -ium emitter
    # crashed sorting a fused numbering with int and '9a' locants (TypeError in
    # ions._lowest_cation_locant_renumbering). The substituent is spelled
    # without locants: '(C6H5)2C2* diphenylmethylidene (PIN)',
    # the Blue Book). OPSIN 2.9.0: full InChIKey and canonical SMILES
    # exact. 'quinolizidine' is not a PIN; this is the best-effort tier.
    ("C[N@+]12CCCC[C@@H]1CCC(=C(C3=CC=CC=C3)C4=CC=CC=C4)C2.[Br-]",
     "(5R,9aR)-3-(diphenylmethylidene)-5-methylquinolizidin-5-ium bromide"),
    ("[B-](/C/1=C/C=C\\C/C=C\\C1)(C2=CC=CC=C2)(C3=CC=CC=C3)C4=CC=CC=C4",
     "(1Z,3Z,6Z)-(cycloocta-1,3,6-trien-1-yl)triphenylboranuide"),
    # D3: the plain C1 methyl is spelled 'methyl' (not 'methan-1-yl');
    # with its inner parens gone the enclosure de-escalates one level ({} ->
    # ), and 'methyl' vs 'methan' flips the alphanumerical order
    # so '6-methyl' now precedes the methyl-bearing complex prefix.
    # RT-verified identical InChIKey (checked below by _validity_gate_name_to_smiles).
    # roadmap N5c/N5d (name-quality lane L2): the amino N roots '[methyl(...)amino]'
    #, the Blue Book) and the ring is 'piperidin-4-yl',
    #:8482). The oxime keeps '(2-oxa-1-azaethan-1-ylidene)' (the plan's residual R5).
    ("CC1=NC(=C(C=C1)/C(=N/O)/N(C)C2CCN(CC2)C)OC3=CC=CC(=C3)C(C)C",
     "(1Z)-6-methyl-3-{[methyl(1-methylpiperidin-4-yl)amino](2-oxa-1-azaethan-1-ylidene)"
     "methyl}-2-[3-(propan-2-yl)phenoxy]pyridine"),
]


@pytest.mark.opsin_gate  # the reclaim lives INSIDE the OPSIN validity gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,expected", RECLAIM_CASES)
def test_stereo_omission_reclaimed_and_roundtrips(smiles, expected):
    """Best-effort names the stereo-complete form AND it full-InChIKey round-trips."""
    with jvm_slots(1, purpose="p2-stereo-reclaim-test"):
        name = _best_effort().name(smiles)
        assert name == expected
        # 0-wrong: the emitted name denotes exactly the input structure.
        out_smi = _validity_gate_name_to_smiles(name)
        assert out_smi is not None
        assert _full_ik(out_smi) == _full_ik(smiles)


@pytest.mark.opsin_gate  # the reclaim lives INSIDE the OPSIN validity gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,expected", RECLAIM_CASES)
def test_disabling_reclaim_abstains(monkeypatch, smiles, expected):
    """Tripwire: with the reclaim off the row does not get the composed name ->
    proves the composition is what emits it (not some other path).

    Suite fix j6: two rows now have a second, RT-exact best-effort name when the
    reclaim is off -- the universal floor names the boranuide (covalent B is in
    its scope since TRIAGE g6 C24) and the general engine names the
    quinolizinium salt as '(1R,6R)-3-(diphenylmethylidene)-1-methyl-1-azabicyclo
    [4.4.0]decan-1-ium bromide' (the salt no longer crashes, TRIAGE g7 C15). So
    the row either abstains, or ships a DIFFERENT name that round-trips to the
    input's full InChIKey (0-wrong)."""
    monkeypatch.setattr(_namer, "_STEREO_OMISSION_RECLAIM", False)
    with jvm_slots(1, purpose="p2-stereo-reclaim-test"):
        name = _best_effort().name(smiles)
        if is_failure_name(name) or "unknown" in name.lower():
            return
        assert name != expected, name
        out_smi = _validity_gate_name_to_smiles(name)
    assert out_smi is not None and _full_ik(out_smi) == _full_ik(smiles), name


def test_retry_cascade_returns_gate_composed_not_flat(monkeypatch):
    """a review -P2 F1 regression: the gate-rejection retry cascade must return the
    gate's OUTPUT, not the pre-gate candidate.

    The stereo-omission reclaim makes ``_final_opsin_validity_gate`` return a name
    DIFFERENT from its input (the flat name with the input's stereo composed back
    on). The cascade loop used to ``return cand`` (the flat, stereo-INCOMPLETE
    name) — a wrong name at the RT-verified tier. It must return the composed name.
    White-box: stub the producer + gate so an alternate dispatch class yields a
    flat name the gate 'composes'.
    """
    from orthonym.errors import _DESCRIPTIVE_FALLBACK_NAMES
    fallback = next(iter(_DESCRIPTIVE_FALLBACK_NAMES))
    flat = "butan-2-ol"
    composed = "(2R)-butan-2-ol"

    eng = _best_effort()
    eng._last_dispatch_class = "stub_alt_class"
    eng._excluded_dispatch_classes = set()
    monkeypatch.setattr(eng, "_name_impl", lambda s: flat)
    monkeypatch.setattr(_namer, "_final_stereo_check",
                        lambda _mol, cand, **kw: cand)
    monkeypatch.setattr(_namer, "_final_grammar_check",
                        lambda cand, *a, **kw: cand)
    # the gate 'composes' the flat name -> returns the stereo-complete form
    monkeypatch.setattr(_namer, "_final_opsin_validity_gate",
                        lambda cand, *a, **kw: composed if cand == flat else cand)

    out = eng._retry_cascade_on_gate_rejection(
        "C[C@@H](O)CC", pre_gate="a-real-suppressed-name", gated=fallback)
    assert out == composed, f"cascade returned {out!r}, expected the composed name"


@pytest.mark.roundtrip
def test_compose_is_deterministic_across_spellings():
    """a review -P2 F2 regression: the composed descriptor must not depend on input
    atom order. All valid SMILES spellings of meso-butane-2,3-diol must compose to
    ONE name (was 2 distinct outputs via the arbitrary first GetSubstructMatch)."""
    from orthonym.namer import _try_compose_input_stereo, _self_consistency_full_key
    spellings = [
        "C[C@@H](O)[C@@H](O)C", "C[C@H](O)[C@H](O)C",
        "O[C@@H](C)[C@@H](C)O", "[C@@H](C)(O)[C@@H](C)O",
    ]
    with jvm_slots(1, purpose="p2-f2-determinism-test"):
        outs = {
            _try_compose_input_stereo(
                "butane-2,3-diol", s, _self_consistency_full_key(s))
            for s in spellings
        }
    assert len(outs) == 1, f"non-deterministic across spellings: {outs}"
    assert None not in outs, "a valid meso spelling failed to compose"
