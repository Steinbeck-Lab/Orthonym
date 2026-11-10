"""A charge-separated S(IV) oxide that gets no name as drawn, named through its S=O drawing.

'R-SO-R′' SULFOXIDES AND SULFONES, the Blue Book) is often drawn charge-separated, ``C[S@+]([O-])c1ccccc1``
for ``C[S@](=O)c1ccccc1``: one compound, one standard InChIKey. The producers of sulfoxides, sulfinyl groups and the
λ4-sulfur rings read the S=O drawing ('(S)-(methanesulfinyl)benzene', '6,7-dihydro-5H-5λ4-thieno[3,2-d]pyrimidin-5-one');
the charged drawing reaches the general engine only, which names some of them as an 'oxido...ium' von Baeyer cation
and no name at all for others (the stereo ones of fused rings). When the drawing as written gets no name,
:func:`sulfoxide_drawing_row` names the S=O drawing: each S+ of three bonds, one a single bond to a terminal O-, gets a
double bond to that oxygen and neither charge. A tag on the sulfur keeps its meaning, as the order of its neighbours
does not change.

The drawing is built from the standard reading of the input's lone-pair centres and shown to be the input
(:func:`s_o_drawing`); the name ships only when OPSIN reads it back to the full standard InChIKey of the input's standard
reading and the lone-pair check of the namer agrees on the drawing (``namer._lone_pair_configuration_verified``).
Its label is systematic_verified at best (never better than the label of the S=O drawing's own name): the name of another
drawing is never certified as the PIN of the drawing as written.
A row the drawing as written names is never replaced, so no name or tier of today changes; the default tier keeps its
decline.
"""
from typing import Optional

from rdkit import Chem

#: a sulfur(IV) oxide drawn charge-separated: S+ with three bonds, one a single bond to a terminal O-
_SITE = Chem.MolFromSmarts("[S+;X3;H0;v3]-[O-;X1]")


def _sites(mol):
    return mol.GetSubstructMatches(_SITE) if mol is not None else ()


def has_charge_separated_sulfoxide(smiles) -> bool:
    """True for a SMILES string with a charge-separated S(IV) oxide (cheap on any other input)."""
    if not isinstance(smiles, str) or '+' not in smiles or 'O-' not in smiles:
        return False
    return bool(_sites(Chem.MolFromSmiles(smiles)))


def _standard_heavy(smiles: str):
    """The heavy-atom Mol of ``smiles`` in the standard reading of its lone-pair centres (the lone pair an implicit
    hydrogen, as ``centres`` and OPSIN read it; ``perception.lone_pair_reading``), each atom with ``_lp_pos``, its
    position in the string. RDKit reads some of these centres the other way ('[O-][S@@+]1CCC...' at the ring-closure
    digit), and a drawing written from its reading would name the mirror image."""
    from .. import namer as _namer
    from ..perception.lone_pair_reading import standard_reading_flips
    heavy = _namer._lp_heavy_mol(smiles)
    if heavy is None:
        return None
    work = Chem.RWMol(heavy)
    for i in standard_reading_flips(smiles):
        work.GetAtomWithIdx(i).InvertChirality()
    return work


def s_o_drawing(smiles: str) -> Optional[str]:
    """The SMILES of ``smiles`` with every charge-separated S(IV) oxide drawn S=O, or None. It is shown to be the
    input: one full standard InChIKey of the two standard readings, and every R/S label ``centres`` gives the input
    string is the label it gives the same atom of the drawing (fail closed on a label of one side only)."""
    from ..perception.centres_bridge import _smiles_output_order
    from ..perception.lone_pair_reading import standard_centres_labels, standard_reading_inchikey
    rw = _standard_heavy(smiles)
    sites = _sites(rw)
    if not sites:
        return None
    for s_idx, o_idx in sites:
        rw.GetAtomWithIdx(s_idx).SetFormalCharge(0)
        rw.GetAtomWithIdx(o_idx).SetFormalCharge(0)
        rw.GetBondBetweenAtoms(s_idx, o_idx).SetBondType(Chem.BondType.DOUBLE)
    try:
        Chem.SanitizeMol(rw)
        out = Chem.MolToSmiles(rw)
        order = _smiles_output_order(rw)
        if (order is None or Chem.MolFromSmiles(out) is None
                or standard_reading_inchikey(out) != standard_reading_inchikey(smiles)):
            return None
        lab_in, lab_out = standard_centres_labels(smiles), standard_centres_labels(out)
    except Exception:  # noqa: BLE001 -- a drawing that cannot be built or shown is no drawing to name
        return None
    if lab_in is None or lab_out is None:
        return None
    mapped = {rw.GetAtomWithIdx(order[pos - 1]).GetIntProp('_lp_pos'): lab
              for pos, lab in lab_out.items() if 0 < pos <= len(order) and lab in ('R', 'S')}
    wanted = {pos: lab for pos, lab in lab_in.items() if lab in ('R', 'S')}
    return out if mapped == wanted else None


def _reads_back(name: str, smiles: str, drawing: str) -> bool:
    """OPSIN reads ``name`` to the full standard InChIKey of the standard reading of ``smiles``, and the lone-pair
    check of the namer agrees on ``drawing``, the S=O drawing:func:`s_o_drawing` showed to be ``smiles`` (the check
    compares centres of one charge form)."""
    from .. import namer as _namer
    from ..perception.lone_pair_reading import standard_reading_inchikey
    opsin_smiles = _namer._validity_gate_name_to_opsin_smiles(name)
    if not opsin_smiles or standard_reading_inchikey(opsin_smiles) != standard_reading_inchikey(smiles):
        return False
    return _namer._lone_pair_configuration_verified(name, drawing)


def sulfoxide_drawing_row(namer, smiles, written: dict) -> Optional[dict]:
    """The ``name_tiered`` row of ``smiles`` named through its S=O drawing, or None: the drawing as written was named
    (``written``), the default tier applies, there is no charge-separated S(IV) oxide, or the S=O drawing gets no name
    that reads back to the input."""
    if written.get('tier') != 'abstain' or namer._default_tier_policy_applies():
        return None
    drawing = s_o_drawing(smiles)
    if drawing is None:
        return None
    row = namer.name_tiered(drawing)
    name = row.get('name')
    if row.get('tier') == 'abstain' or not name or not _reads_back(name, smiles, drawing):
        return None
    from .decline_fallback import capped
    return capped({**row, 'tier': 'systematic_verified', 'is_pin': False, 'source': 'sulfoxide_drawing',
                   's_o_drawing': drawing}, row)
