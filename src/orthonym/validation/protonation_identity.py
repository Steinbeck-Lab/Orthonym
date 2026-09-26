"""Protonation-site identity: does a name's OPSIN parse carry its hydrons on the
same atoms as the input structure?

The standard InChI moves every hydron of a protonated (onium) atom into one
mobile /p layer, so the full standard InChIKey cannot tell WHICH atom carries
the charge. Protonation isomers of one cation share one key:

    C[NH2+]CCC(=O)NC vs CNCCC(=O)[NH2+]C (aminium vs amidium)
    C[NH2+]CCC#N vs CNCCC#[NH+] (aminium vs nitrilium)
    C[NH+](C)CCN vs CN(C)CC[NH3+] (tertiary vs primary aminium)

Every gate that accepts a name on full-key equality alone is therefore blind to
a name that puts the '-ium' on the wrong nitrogen ('N-methyl-3-(methylamino)-
propanamidium' for the protonated amine). (the Blue Book) and
Table 7.4 (:41417-41424) define 'amidium', 'nitrilium', 'aminium' as the
cationic form OF that characteristic group, so such a name denotes a different
species.

The stricter rule, applied only when BOTH the input and the parse carry a
protonated heavy atom (a positively charged non-hydrogen atom bearing hydrogen):
the fixed-hydrogen InChI (``/FixedH``, stereo layers off) must be equal. The
fixed-H layer records where each hydron and charge sits, and InChI's own
normalisation keeps charge-delocalised forms equal (the two resonance drawings
of an amidinium or a 4-aminopyridinium give one fixed-H InChI), so a correct
name is never rejected for drawing the charge on the other resonance atom.

Scoped to "both sides" on purpose: a salt drawn as its ionic pair and named as
the neutral acid-base form ('...amine hydrochloride', a proton on Cl), or an
amino-acid zwitterion named as the neutral amino acid,
the Blue Book), is not a protonation-SITE question and keeps its existing
treatment; those are judged by the callers' own charge checks.
"""
from rdkit import Chem
from rdkit.Chem import inchi as _inchi


def _has_protonated_heavy_atom(mol) -> bool:
    return any(a.GetAtomicNum() > 1 and a.GetFormalCharge() > 0 and a.GetTotalNumHs() > 0
               for a in mol.GetAtoms())


def _fixed_h_inchi(mol):
    try:
        return _inchi.MolToInchi(mol, options="/FixedH /SNon")
    except Exception:
        return None


def protonation_site_verdict(input_smiles: str, parsed_smiles: str) -> str:
    """``"n/a"`` unless both structures carry a protonated heavy atom, else
    ``"ok"`` (the fixed-H InChIs are equal) or ``"mismatch"``. An input or parse
    RDKit cannot read, or an InChI that cannot be computed, is ``"n/a"``: the
    callers' own checks decide those."""
    mi = Chem.MolFromSmiles(input_smiles) if input_smiles else None
    mo = Chem.MolFromSmiles(parsed_smiles) if parsed_smiles else None
    if mi is None or mo is None:
        return "n/a"
    if not (_has_protonated_heavy_atom(mi) and _has_protonated_heavy_atom(mo)):
        return "n/a"
    fi, fo = _fixed_h_inchi(mi), _fixed_h_inchi(mo)
    if not fi or not fo:
        return "n/a"
    return "ok" if fi == fo else "mismatch"
