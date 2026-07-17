"""v24 W8 P3 — amino-acid retained-name derivatives (P-103).

Accuracy-first scope: only the derivative classes whose retained PIN form is
UNAMBIGUOUS in the Blue Book are built here. The substituted-derivative retained
forms (5-hydroxytryptophan, N6-acetyl-lysine, O-phospho-serine, hydroxyproline
base) are DEFERRED — P-103 carries no `(PIN)` markers, several BB examples list
the *systematic* name first, and the project's existing golds already name
substituted/non-standard amino acids systematically. Emitting a retained form
where the systematic is the PIN would regress a correct name, so those keep the
(already-correct) systematic output.

Built classes:
  * 3.1 `allo` diastereomers of threonine/isoleucine (P-103.1.3.2.2 — 'allo' is a
        MANDATED modifier of the retained name; threonine/isoleucine are PIN
        retained names, so their C-3 epimer's PIN is `allo-<name>`). The map also
        covers the D-/D-allo- forms (all 8 OPSIN-RT verified).

Deferred (documented, accuracy-first — do NOT emit a retained form where the PIN
is uncertain, to avoid regressing a correct systematic name):
  * 3.2 esters (`methyl L-alaninate`): the retained-stem is the PIN by the
        `methyl acetate` functional-parent analogy, BUT the exact string is
        entangled with the unresolved bare-AA stereo-descriptor policy (the
        project emits bare `alanine` with L IMPLICIT, while BB 54601 shows explicit
        `L-alaninate`; a "Stereo backstop" already flags bare `alanine` as
        under-specified). Resolve the L-implicit-vs-explicit policy first.
  * 3.3-3.6 substituted-derivative retained forms (5-hydroxytryptophan,
        N6-acetyl-lysine, O-phospho-serine, hydroxyproline base): P-103 carries NO
        `(PIN)` markers, several P-103.2.3 examples list the systematic name FIRST,
        and the project's existing golds name substituted/non-standard AAs
        systematically (e.g. S-ethylcysteine -> 2-amino-3-(ethylsulfanyl)propanoic
        acid). Keeping the (already-correct) systematic avoids a wrong-PIN regression.
  * 3.7 non-standard zwitterion: the current neutral-form output is SANCTIONED by
        P-103.2.4.1 (naming the conventional neutral form of a monoamino
        monocarboxylic acid) and passes the project's charge-normalized RT gate, so
        it is a valid name, not a leak; failing it closed would regress it.
"""
import pytest
from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit

G = Orthonym()


# --- 3.1 allo diastereomers (P-103.1.3.2.2) -------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # L-allo-threonine = (2S,3S) [CIP-verified]; C-3 epimer of L-Thr (2S,3R).
    ("C[C@H](O)[C@H](N)C(=O)O", "allo-threonine"),
    # L-allo-isoleucine = (2S,3R) [CIP-verified]; C-3 epimer of L-Ile (2S,3S).
    ("CC[C@@H](C)[C@H](N)C(=O)O", "allo-isoleucine"),
])
def test_allo_amino_acids(smiles, expected):
    assert G.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # All 8 stereoisomers of Thr/Ile emit their retained-name PIN (OPSIN-RT verified).
    ("C[C@@H](O)[C@H](N)C(=O)O", "threonine"),        # L-Thr (2S,3R)
    ("C[C@H](O)[C@@H](N)C(=O)O", "D-threonine"),       # D-Thr (2R,3S)
    ("C[C@@H](O)[C@@H](N)C(=O)O", "D-allo-threonine"), # D-allo-Thr (2R,3R)
    ("CC[C@H](C)[C@H](N)C(=O)O", "isoleucine"),        # L-Ile (2S,3S)
    ("CC[C@@H](C)[C@@H](N)C(=O)O", "D-isoleucine"),    # D-Ile (2R,3R)
    ("CC[C@H](C)[C@@H](N)C(=O)O", "D-allo-isoleucine"),# D-allo-Ile (2R,3S)
])
def test_thr_ile_stereoisomers(smiles, expected):
    assert G.name(smiles) == expected
