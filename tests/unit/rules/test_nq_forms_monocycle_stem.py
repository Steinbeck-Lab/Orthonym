"""Lane L2 proper fix: the monocycle form builds its parent from the structure and its own
numbering; it does not read the shared namer's output, the Blue Book;
,:8470)."""
import pytest
from rdkit import Chem

from orthonym.rules.monocycle_forms import monocycle_form

PARENTS = [
    ("c1ccncc1", "pyridine"), ("C1C=CC=CO1", "2H-pyran"), ("c1c[nH]cn1", "1H-imidazole"),
    ("c1cocn1", "1,3-oxazole"), ("C1CCOC1", "oxolane"), ("C1CCNCC1", "piperidine"),
    ("C1COCCN1", "morpholine"), ("C1OCCO1", "1,3-dioxolane"),
    ("C1CCOC=C1", "3,4-dihydro-2H-pyran"), ("c1nc[nH]n1", "1H-1,2,4-triazole"),
    ("c1ccsc1", "thiophene"), ("c1cnoc1", "1,2-oxazole"), ("C1CNC1", "azetidine"),
    ("C1CO1", "oxirane"), ("C1N=NN=N1", "5H-tetrazole"), ("c1nn[nH]n1", "2H-tetrazole"),
    ("c1nnn[nH]1", "1H-tetrazole"), ("C1N=N1", "3H-diazirine"), ("C1OO1", "dioxirane"),
    ("c1cn[nH]n1", "2H-1,2,3-triazole"), ("C1=CCNC=C1", "1,2-dihydropyridine"),
    ("O1CCOCC1", "1,4-dioxane"), ("c1ncncn1", "1,3,5-triazine"), ("C1CSCCS1", "1,4-dithiane"),
    ("c1cc[nH]c1", "1H-pyrrole"), ("C1CC=CN1", "2,3-dihydro-1H-pyrrole"),
]


@pytest.mark.parametrize("smiles,parent", PARENTS)
def test_the_form_never_reads_the_namers_output(monkeypatch, smiles, parent):
    import orthonym.rules.heterocycles as heterocycles

    def refuse(*a, **k):
        raise AssertionError("monocycle_form read the namer's output")
    monkeypatch.setattr(heterocycles, "name_heterocycle", refuse)
    m = Chem.MolFromSmiles(smiles)
    form = monocycle_form(m, list(m.GetRingInfo().AtomRings()[0]))
    assert form is not None and form.parent == parent
