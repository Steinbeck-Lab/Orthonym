"""Task M1 (v51): extend best-effort organometallic + simple-inorganic naming.

User-requested (2026-09-15): molecules like ``C[U]`` (methyluranium), ``C[Mg]C``
(dimethylmagnesium) and ``[O][O]`` (dioxygen) used to ABSTAIN, while sibling
forms (``C[Zn]C`` -> dimethylzinc, ``C[Mg]Cl`` -> methylmagnesium chloride)
already named. Organometallics are (no PIN derivation), so these are
BEST-EFFORT names: emitted ONLY when they OPSIN round-trip (``-r`` radicals-on)
to the input structure, else abstain (0-wrong ABSOLUTE).

Two root causes fixed:
  1. rules/organometallics.py Branch B (metal_direct) hard-coded a per-metal
     ``ligand_class`` dispatch whose ``else: return None`` dropped every metal
     outside {Li,Na,K,Zn,Cd,Hg,Al,Mg}. A sigma-only complex on any metal_direct
     metal now builds ``{multiplied-prefix}{metal}`` (no Stock -- every existing
     metal_direct row is stock-free, so the existing forms are byte-identical),
     gated downstream by the OPSIN validity gate.
  2. data/organometallics.py METAL_NAMES lacked Group-2 (Sr/Ba/Ra), Group-3
     (Sc/Y/La/Ac), the lanthanides and the actinides (incl. U). Added as
     metal_direct (every ``methyl<metal>`` verified OPSIN-`-r` round-trips).
  3. data/retained_names.py keyed dioxygen by the ``O=O`` canonical SMILES only;
     the diradical spelling ``[O][O]`` (a distinct RDKit-canonical form, same
     InChIKey) missed the lookup. Added the ``[O][O]`` key.
"""

import pytest

from orthonym import name_compound, errors

try:
    from rdkit import Chem
    _RDKIT = True
except Exception:  # pragma: no cover
    _RDKIT = False

from orthonym.jvm_bridge import opsin_available, opsin_stdout


# (input SMILES, expected best-effort name) -- names ABSTAINED before M1.
_TARGETS = [
    ("C[U]", "methyluranium"),
    ("C[Mg]C", "dimethylmagnesium"),
    ("CC[Mg]CC", "diethylmagnesium"),
    ("C[Ca]C", "dimethylcalcium"),
    ("C[Be]C", "dimethylberyllium"),
    ("C[Th]", "methylthorium"),
    ("[O][O]", "dioxygen"),
]

# Positive controls -- named BEFORE M1, must be byte-identical AFTER.
_CONTROLS = [
    ("C[Zn]C", "dimethylzinc"),
    ("CC[Zn]CC", "diethylzinc"),
    ("C[Mg]Cl", "methylmagnesium chloride"),
    ("C[Al](C)C", "trimethylaluminum"),
    ("C[Na]", "methylsodium"),
    ("C[Hg]C", "dimethylmercury"),
    ("O=O", "dioxygen"),
]

# Hard tail -- must STAY abstained (0-wrong: never forced to a wrong name).
# * the bridged bimetallic acetylide (2 Mg, fused rings) is multinuclear ->
# out of Phase-161 scope (is_multimetal);
# * MgCl2 (halide-only, no organic sigma-ligand) has no best-effort form here.
_HARD_TAIL = [
    "C1#C[Mg]C#CC2OC3C#C[Mg]C#CC=3OC1=2",
    "Cl[Mg]Cl",
]


@pytest.mark.parametrize("smiles,expected", _TARGETS)
def test_target_names(smiles, expected):
    """Each M1 target now emits its best-effort name (was an abstention)."""
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles,expected", _CONTROLS)
def test_controls_unchanged(smiles, expected):
    """Every pre-M1 correct organometallic/diatomic name is preserved."""
    assert name_compound(smiles) == expected


@pytest.mark.parametrize("smiles", _HARD_TAIL)
def test_hard_tail_abstains(smiles):
    """The hard tail must abstain -- never forced to a wrong name."""
    out = name_compound(smiles)
    assert errors.is_failure_name(out), f"expected abstention, got {out!r}"


@pytest.mark.roundtrip
@pytest.mark.skipif(not _RDKIT, reason="RDKit required")
@pytest.mark.parametrize("smiles,expected", _TARGETS + _CONTROLS)
def test_target_round_trips_via_opsin_radicals(smiles, expected):
    """0-wrong: every emitted name OPSIN-`-r` round-trips to the input skeleton
    InChIKey (the shipped-oracle path). If OPSIN is unavailable the naming-side
    validity gate would itself have fail-OPENed, so the RT proof is JVM-gated."""
    if not opsin_available():
        pytest.skip("OPSIN JVM not available")
    name = name_compound(smiles)
    assert name == expected
    osmi, served = opsin_stdout(name, allow_radicals=True)
    assert served and osmi and osmi not in ("", "REJECTED"), (
        f"OPSIN did not parse {name!r}"
    )
    in_mol = Chem.MolFromSmiles(smiles)
    out_mol = Chem.MolFromSmiles(osmi)
    assert in_mol is not None and out_mol is not None
    in_key = Chem.MolToInchiKey(in_mol)[:14]
    out_key = Chem.MolToInchiKey(out_mol)[:14]
    assert in_key == out_key, (
        f"{name!r} round-trips to a DIFFERENT skeleton: "
        f"in={in_key} opsin={out_key} ({osmi})"
    )
