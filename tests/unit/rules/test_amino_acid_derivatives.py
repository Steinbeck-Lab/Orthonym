""" W8 P3 — amino-acid retained-name derivatives.

Accuracy-first scope: only the derivative classes whose retained PIN form is
UNAMBIGUOUS in the Blue Book are built here. The substituted-derivative retained
forms (5-hydroxytryptophan, N6-acetyl-lysine, O-phospho-serine, hydroxyproline
base) are DEFERRED — carries no `(PIN)` markers, several BB examples list
the *systematic* name first, and the project's existing golds already name
substituted/non-standard amino acids systematically. Emitting a retained form
where the systematic is the PIN would regress a correct name, so those keep the
(already-correct) systematic output.

Built classes:
  * 3.1 `allo` diastereomers of threonine/isoleucine — 'allo' is a
        MANDATED modifier of the retained name; threonine/isoleucine are PIN
        retained names, so their C-3 epimer's PIN is `allo-<name>`). The map also
        covers the D-/D-allo- forms (all 8 OPSIN-RT verified).
  * 3.2 esters, BB 54595-54608): `methyl <descriptor><name>ate` for the
        single-alpha-stereocentre monocarboxylic standard AAs + glycine. The
        L-implicit-vs-explicit policy is now RESOLVED for esters: BB 54601 shows
        `methyl L-alaninate` with an EXPLICIT descriptor (unlike the bare AA, which
        suppresses implicit L) -- so the ester stem always cites L-/D- explicitly
        (glycine, achiral, gets none). Diacid AAs (aspartic/glutamic -- need
        positional ester locants) and 2-stereocentre AAs (threonine/isoleucine --
        allo entanglement) are OUT of scope and fall through to the pre-existing
        systematic ester name (no regression).
  * 3.7 non-standard-AA zwitterion /, BB 54554-54569):
        the "convenient neutral form" dispensation is licensed ONLY
        for the monoamino monocarboxylic acids retained in Table 10.4 (the 20
        canonical STANDARD_AMINO_ACIDS). BB's own example
        (S-methyl-L-cysteine zwitterion) shows the PIN for a NON-standard amino-acid
        zwitterion is the Method-1 IONIC form -- the anion is the parent,
        the protonated amine an `azaniumyl` prefix.
        UPDATED (charged Slice B, `charged_router` GUARD 4 ->
        `_name_primary_amine_azaniumyl_zwitterion`): Orthonym now BUILDS that
        ionic PIN on the DEFAULT path -- `(2R)-2-azaniumyl-3-(methylsulfanyl)-
        propanoate` -- correcting the intermediate 4782742f over-reach that shipped
        the NON-PIN neutral `(2R)-2-amino-3-(methylsulfanyl)propanoic acid` on the
        PIN path: a zwitterion's neutral form is not its PIN). The
        alpha stereocentre survives because the builder neutralizes IN PLACE and
        re-expresses the amino prefix (it does not sever/cap the stereocentre), and
        every emission is full-InChIKey RT-gated (charges + stereo). The neutral
        systematic name remains reachable on the BEST-EFFORT tier only. Standard-AA
        zwitterions (glycine/L-alanine/...) are UNCHANGED (still the retained name,
        fired before route_charged).

Deferred (documented, accuracy-first — do NOT emit a retained form where the PIN
is uncertain, to avoid regressing a correct systematic name):
  * 3.3-3.6 substituted-derivative retained forms (5-hydroxytryptophan,
        N6-acetyl-lysine, O-phospho-serine, hydroxyproline base): carries NO
        `(PIN)` markers, several examples list the systematic name FIRST,
        and the project's existing golds name substituted/non-standard AAs
        systematically (e.g. S-ethylcysteine -> 2-amino-3-(ethylsulfanyl)propanoic
        acid). Keeping the (already-correct) systematic avoids a wrong-PIN regression.
"""
import pytest
from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit

G = Orthonym()
RAW = Orthonym(_disable_opsin_validity_gate=True)  # gate-off: proves the raw namer


# --- 3.1 allo diastereomers -------------------------------
# -REGRESSION I12: the L forms below expected the BARE retained name, because
# the producer suppressed 'L-'. That suppression is `### **** Indication of
# configuration in peptides` (the Blue Book) applied outside its scope — and
# `## **** Use of the prefix 'allo'` (:54320), the section this pack was
# built on, writes all four forms WITH the descriptor at:54324-54330:
# L-isoleucine (symbols 'Ile',' I') (2S,3S)-2-amino-3-methylpentanoic acid
# L-alloisoleucine (symbol 'aIle') (2S,3R)-2-amino-3-methylpentanoic acid
# L-threonine (symbols 'Thr','T') (2S,3R)-2-amino-3-hydroxybutanoic acid
# L-allothreonine (symbol 'aThr') (2S,3S)-2-amino-3-hydroxybutanoic acid
# The D rows are unchanged, which is the point: only the L was being lost.
@pytest.mark.parametrize("smiles,expected", [
    # L-allothreonine = (2S,3S) [CIP-verified]; C-3 epimer of L-Thr (2S,3R).
    ("C[C@H](O)[C@H](N)C(=O)O", "L-allothreonine"),
    # L-alloisoleucine = (2S,3R) [CIP-verified]; C-3 epimer of L-Ile (2S,3S).
    ("CC[C@@H](C)[C@H](N)C(=O)O", "L-alloisoleucine"),
])
def test_allo_amino_acids(smiles, expected):
    assert G.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # All 8 stereoisomers of Thr/Ile emit their retained name (OPSIN-RT verified).
    # These are retained names, NOT PINs: `### ** INTRODUCTION**` (:50943) —
    # "Preferred IUPAC names (PINs) are not identified for the compounds in this
    # Chapter."
    ("C[C@@H](O)[C@H](N)C(=O)O", "L-threonine"),      # L-Thr (2S,3R)
    ("C[C@H](O)[C@@H](N)C(=O)O", "D-threonine"),       # D-Thr (2R,3S)
    ("C[C@@H](O)[C@@H](N)C(=O)O", "D-allothreonine"), # D-allo-Thr (2R,3R)
    ("CC[C@H](C)[C@H](N)C(=O)O", "L-isoleucine"),      # L-Ile (2S,3S)
    ("CC[C@@H](C)[C@@H](N)C(=O)O", "D-isoleucine"),    # D-Ile (2R,3R)
    ("CC[C@H](C)[C@@H](N)C(=O)O", "D-alloisoleucine"),# D-allo-Ile (2R,3S)
])
def test_thr_ile_stereoisomers(smiles, expected):
    assert G.name(smiles) == expected


# --- 3.2 amino-acid esters, BB 54595-54608) --------------------
@pytest.mark.parametrize("smiles,expected", [
    # BB 54601 verbatim example. (The parenthetical "unlike the bare AA's implicit L"
    # was removed in -REGRESSION I12: the bare AA now carries its L too.)
    ("COC(=O)[C@H](C)N", "methyl L-alaninate"),
    # Glycine is achiral -- no descriptor.
    ("COC(=O)CN", "methyl glycinate"),
    # Different R' (ethyl) -- reuses the existing alkyl namer, not hardcoded.
    ("CCOC(=O)[C@H](C)N", "ethyl L-alaninate"),
    # D- enantiomer (alpha-R): explicit D-.
    ("COC(=O)[C@@H](C)N", "methyl D-alaninate"),
])
def test_amino_acid_esters(smiles, expected):
    assert RAW.name(smiles) == expected


@pytest.mark.parametrize("smiles", [
    # Diacid AAs (aspartic/glutamic) need positional ester locants
    # (`1-methyl L-aspartate`) -- out of scope, must NOT regress to a wrong name.
    "COC(=O)[C@@H](N)CC(=O)O",
    # 2-stereocentre AAs (threonine/isoleucine) -- allo entanglement, deferred.
    "COC(=O)[C@H](N)[C@@H](C)O",
])
def test_amino_acid_esters_out_of_scope_falls_through(smiles):
    # Must not be "unknown" and must not silently drop atoms/charges -- the
    # pre-existing systematic ester path still names it (no regression), it
    # just doesn't get the retained-stem treatment.
    name = RAW.name(smiles)
    assert name is not None
    assert "unknown" not in name


# --- 3.7 non-standard-AA zwitterion: azaniumyl ionic PIN /
#, charged Slice B) ---------------------------------------
@pytest.mark.opsin_gate
def test_non_standard_zwitterion_builds_azaniumyl_pin():
    # S-methylcysteine-family zwitterion: BB's OWN example (S-methyl-L-cysteine
    # zwitterion, shows the PIN for a NON-standard amino-acid
    # zwitterion is the Method-1 IONIC form -- anion is the parent,
    # the protonated amine is an `azaniumyl` prefix. charged Slice B now
    # BUILDS it on the DEFAULT path (route_charged GUARD 4 ->
    # `_name_primary_amine_azaniumyl_zwitterion`), correcting the 4782742f
    # over-reach that shipped the NON-PIN neutral `...propanoic acid` here.
    # The alpha stereocentre survives (neutralize-in-place, not sever), so the
    # descriptor is spelled: the builder full-InChIKey RT-gates its own
    # emission, so this is 0-wrong (the input's InChIKey
    # an InChIKey -- (2R), NOT (2S) as an early note guessed --
    # is reproduced by OPSIN-parsing the name). RAW (gate off) proves the
    # producer's OWN RT gate, not the namer's.
    from rdkit import Chem
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smi = "CSC[C@H]([NH3+])C(=O)[O-]"
    name = RAW.name(smi)
    assert name == "(2R)-2-azaniumyl-3-(methylsulfanyl)propanoate"
    # must NOT be the non-PIN neutral acid on the default path
    assert "propanoic acid" not in name
    g = opsin_parse(name)
    assert g and Chem.MolToInchiKey(Chem.MolFromSmiles(g)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


@pytest.mark.parametrize("smiles,expected", [
    # Standard-AA zwitterions (Table 10.4, dispensation) are
    # UNCHANGED by the veto.
    ("[NH3+]CC(=O)[O-]", "glycine"),
    ("C[C@H]([NH3+])C(=O)[O-]", "L-alanine"),
])
def test_standard_aa_zwitterion_unaffected(smiles, expected):
    assert RAW.name(smiles) == expected
    assert G.name(smiles) == expected
