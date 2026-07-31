"""v29 Phase 8: a bridging diaryl/aryl-heteroaryl amine must name deterministically.

Two interlocking defects, both reproduced on ``Brc1ccccc1Nc1ccccn1``:

A. **Order-dependent parent selection.** ``get_principal_group`` normalises an
   amine PCG match to a ``(N, bearing-C)`` 2-tuple
   (``seniority._normalize_pcg_match``), keeping ONE of the N's carbons --
   tie-broken by ATOM INDEX. For an amine N that BRIDGES two rings the two
   candidate carbons tie, so ``is_principal_group_on_ring`` reported the group on
   whichever ring happened to be spelled first. When the survivor sat in the
   JUNIOR ring, ``pg_on_senior`` went False at ``namer.py:4337`` and the P-44.2
   senior ring (pyridine) was discarded for ``atom_rings[0]`` (benzene). Measured
   at ``b0e5c3af``: 8 of 12 randomised orderings named the molecule, 4 abstained
   after the OPSIN validity gate suppressed the malformed
   ``'bromo-N-pyridylanamine'`` that the fallback amine assembler produced.

B. **Missing enclosing marks.** Even on the orderings that named it, the italic-N
   compound prefix was cited naked: ``N-2-bromophenylpyridin-2-amine``.
   P-16.5.1.1 requires parentheses around a compound prefix.

Governing rules, quoted with their section headings:

* **P-16.5 ENCLOSING MARKS** -> **P-16.5.1 Parentheses (also called curves or
  round brackets)** -> **P-16.5.1.1**: "Parentheses are used around compound (see
  P-29.1.2) and complex (see P-29.1.3) prefixes; after the multiplicative
  prefixes 'bis', 'tris', etc.; ..." Its first example is exactly this shape --
  a SINGLE, unmultiplied compound prefix: ``Cl-CH2-SiH3`` ->
  ``(chloromethyl)silane`` (PIN).
* **P-29.1.2 "A compound substituent group"**: "A compound substituent group
  consists of a simple substituent group (the parent substituent group) to which
  is attached one or more simple substituent groups." ``2-bromophenyl`` is
  ``phenyl`` bearing ``bromo``, so it is a compound prefix.

NOTE: P-16.3.4 is *not* the governing rule here despite its title
("Parentheses (round brackets)"). Its own opening sentence limits it to
MULTIPLIED components -- "Parentheses (round brackets) (see P-16.6.1) are used to
enclose *multiplied* components that are: (a) simple substituent prefixes having
locants" -- and it sits under the heading "P-16.3 Multiplicative prefixes 'di',
'tri', etc. vs. 'bis', 'tris', etc." There is a single 2-bromophenyl here.
"""

import random

import pytest
from rdkit import Chem

from orthonym import name_compound

BRIDGING_SMILES = "Brc1ccccc1Nc1ccccn1"
EXPECTED = "N-(2-bromophenyl)pyridin-2-amine"


def _randomised_smiles(base: str, n: int, seed: int):
    """Distinct atom orderings of ``base``, written non-canonically."""
    mol = Chem.MolFromSmiles(base)
    assert mol is not None, f"fixture SMILES did not parse: {base}"
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        out.append(
            Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        )
    return out


def test_bridging_aryl_heteroaryl_amine_name_is_the_pin():
    """P-44.2 picks the pyridine parent; P-16.5.1.1 encloses the compound prefix."""
    assert name_compound(BRIDGING_SMILES) == EXPECTED


def test_bridging_amine_name_is_stable_across_atom_orderings():
    """The regression is a NONDETERMINISM: one ordering could never catch it.

    At ``b0e5c3af`` this split 8 named / 4 abstained over these same 12 orderings.
    """
    orderings = _randomised_smiles(BRIDGING_SMILES, 12, seed=20260731)
    assert len(orderings) == 12, "harness produced no orderings"

    names = {}
    for smi in orderings:
        names[smi] = name_compound(smi)

    assert len(names) >= 2, (
        "randomisation collapsed to a single spelling -- the test would be "
        f"vacuous; got {list(names)}"
    )
    distinct = set(names.values())
    assert distinct == {EXPECTED}, (
        "atom input order changed the emitted name: "
        + "; ".join(f"{s} -> {n!r}" for s, n in sorted(names.items()))
    )


def test_no_malformed_anamine_fallback_is_emitted():
    """'anamine' is a parent stem that went missing -- never a nomenclature term.

    The suppressed string was ``bromo-N-pyridylanamine``: an empty parent stem
    plus an unlocanted ``bromo``. Guard the whole shape, not the one literal.
    """
    orderings = _randomised_smiles(BRIDGING_SMILES, 12, seed=20260731)
    assert orderings, "harness produced no orderings"
    for smi in orderings:
        name = name_compound(smi)
        assert "anamine" not in name, f"{smi} -> {name!r}"
        assert "pyridylan" not in name, f"{smi} -> {name!r}"


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # The pubchem_2000 row that carried defect B independently of the
        # determinism bug (it named on every ordering, always unenclosed).
        (
            "C1CCN(CC1)C2=NC=NC(=C2)NC3=CC=C(C=C3)Br",
            "N-(4-bromophenyl)-6-(piperidin-1-yl)pyrimidin-4-amine",
        ),
    ],
)
def test_compound_italic_n_prefix_is_enclosed(smiles, expected):
    """P-16.5.1.1 on a second, independent molecule from the corpus."""
    assert name_compound(smiles) == expected


def test_simple_italic_n_prefix_stays_bare():
    """Guard the other side: a SIMPLE prefix must NOT gain parentheses.

    P-16.5.1.1 encloses compound and complex prefixes only; ``methyl`` and
    ``phenyl`` are simple (P-29.1.1), so ``N-methyl...`` / ``N-phenyl...`` are
    correct as they stand. Without this, a too-broad enclosure fix would read as
    a pass.
    """
    assert name_compound("CNc1ccccn1") == "N-methylpyridin-2-amine"
    assert name_compound("c1ccccc1Nc1ccccn1") == "N-phenylpyridin-2-amine"
