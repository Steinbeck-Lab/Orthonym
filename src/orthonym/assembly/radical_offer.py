"""The last resort of the best-effort tier for a molecule with radical centres.

A molecule with a radical centre that no other path of the best-effort tier names is named by the floor with
every centre cited in the name, and the name ships when OPSIN (``-r``) reads it back to the input: its full
InChIKey, its radical graph (``validation.radical_identity``) and its stereo. It is an offer, not a return (a project rule), made only when every other path of the tier, the other offers of ``assembly/default_tier_offer.py`` included,
has left the molecule without a name, so no name that ships changes: a molecule that has a name today keeps it,
whatever the radical path would have given.

What the floor cites in the name, inside:func:`orthonym.validation.read_back_policy.radical_last_resort` and nowhere
else (``radical_floor.radical_floor_names``):

* a divalent or trivalent centre by the lambda convention, "The λ-convention" (the Blue Book, sentence
  :40562): "Divalent and trivalent radical centers in a parent hydride formally derived by the removal of two or three
  hydrogen atoms from the same skeletal atom in its standard valence state may be described by the λ-convention (see
  .... This method is only for general nomenclature." The row is ``systematic_verified``, never a PIN;
* several monovalent centres as the free valences of the one ring system or acyclic chain that holds them all,
  "CHOICE OF PARENT RADICAL" (the Blue Book) with "Multiple radical centers (polyradicals)" (:40524);
* a monovalent centre that no parent carries has no citation: the molecule keeps its abstention (never a lambda on a
  monovalent centre, and the book's 'ylo' of,:40753, is not read by OPSIN).

The stereo of the input is composed on the name of the floor as the stereo-composed offer does it
(``stereo_composed_offer._verified_composed_name``), with the name read by OPSIN's ``-r``
(``opsin_roundtrip.opsin_atom_locant_map``, ``opsin_roundtrip_check``), and a name that cites a locant twice in its
descriptors ships only when the descriptors at that locant are equal and state every stereo unit of the input
(``stereo_composed_offer.repeated_locants_are_one_structure``). A name that states fewer stereo elements than the
input carries the row's ``stereo_unexpressed`` flag, as the name of any other producer does.

The offer is made once, for the molecule the caller asked about: not inside the naming of the stereo-free structure
of the stereo-composed offer (:func:`suspended`), where the structure would be named twice.
"""
import contextlib
import contextvars
from typing import Dict, Optional

#: True while the stereo-free structure of the stereo-composed offer is named: the radical offer is made for the
#: input, after that offer, and not for its stereo-free structure
_SUSPENDED: contextvars.ContextVar = contextvars.ContextVar("orthonym_radical_offer_suspended", default=False)


@contextlib.contextmanager
def suspended():
    """The block names a structure on its way to another offer: the radical offer makes no offer in it."""
    token = _SUSPENDED.set(True)
    try:
        yield
    finally:
        _SUSPENDED.reset(token)


def radical_last_resort_row(namer, smiles) -> Optional[Dict]:
    """The row ``namer`` offers a molecule with radical centres that it left without a name, or None.
    ``smiles`` is a SMILES string or an RDKit molecule, as for ``name_tiered``. Best-effort tier only. Never raises
    (the abstention stands)."""
    try:
        from ..validation.read_back_policy import radical_last_resort_active
        if _SUSPENDED.get() or radical_last_resort_active():
            return None
        from .stereo_composed_offer import _eligible
        if not _eligible(namer):
            return None
        from rdkit import Chem

        from ..input_mol import handles, smiles_for
        if not isinstance(smiles, str) and handles(smiles):
            smiles = smiles_for(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetStereoGroups() or not any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            return None
        return _radical_row(namer, smiles, mol)
    except Exception:                       # the offer must never break a naming: the abstention stands
        return None


def _radical_row(namer, smiles: str, mol) -> Optional[Dict]:
    from ..errors import is_failure_name
    from ..metrics import provenance as pv
    from ..metrics.abstention import peek_abstention, restore_abstention
    from ..metrics.producer_trail import carried
    from ..validation.read_back_policy import radical_last_resort, reading_for
    from ..validation.stereo_stated import name_states_fewer_stereo
    from .default_tier_offer import general_tier_engine
    from .radical_floor import radical_floor_names
    kept_provenance, kept_abstention = pv.get_provenance(), peek_abstention()
    row = None
    try:
        engine = general_tier_engine(namer)
        with radical_last_resort(), reading_for(mol):
            for floor_name in radical_floor_names(mol):
                name = _verified_name(smiles, floor_name)
                if name is None or is_failure_name(name):
                    continue
                pv.clear_provenance()
                pv.record_source("general_engine")
                pv.record_gate_outcome(pv.GATE_OUTCOME_FULL_KEY_VERIFIED, name)
                if carried(name, engine._exit_round_trip_check(name, smiles)) != name:
                    continue                # the exit check every other name passes (ambiguity, lone pair)
                candidate = engine._tier_row_and_pin_form(smiles, name, pv.get_provenance())[0]
                if candidate.get("name") != name or candidate.get("tier") in (None, "abstain") or candidate.get("is_pin"):
                    continue
                candidate["producer"] = "engine:t4"
                if name_states_fewer_stereo(smiles, name) or pv.get_provenance().get("stereo_unexpressed"):
                    candidate["stereo_unexpressed"] = True
                row = candidate
                return row
        return None
    finally:
        # the floor named the molecule on the side: what the caller reads about its abstention is what it was,
        # unless a name is offered
        restore_abstention(kept_abstention)
        if row is None:
            pv.restore_provenance(kept_provenance)


def _verified_name(smiles: str, name: str) -> Optional[str]:
    """``name`` (the floor's), or ``name`` with the stereo of ``smiles`` composed on it, when it reads back to the
    input: OPSIN's read-back is the input's full InChIKey, its radical graph and its hydron site
    (``namer._shipped_name_round_trip``) and, for a lone-pair stereocentre, its configuration; the descriptors of the
    name cite a locant twice only when that is one structure (``repeated_locants_are_one_structure``). The read-back
    is the exact isomeric comparison of canonical SMILES, which sees E/Z (``Orthonym._rt_match``), for a name that
    states the stereo of the input; a name that leaves out stereo the key cannot see (the E/Z of a bond that is single
    in another form) ships when what it states agrees with the input (the per-element comparison), and its row says
    that it states fewer stereo elements than the input. None otherwise."""
    from ..namer import (
        Orthonym, _has_lone_pair_stereocentre, _lone_pair_as_written, _lone_pair_configuration_verified,
        _shipped_name_round_trip, _validity_gate_name_to_smiles)
    from .stereo_composed_offer import _verified_composed_name, repeated_locants_are_one_structure, stereo_free_smiles

    def reads_back(candidate: str, omission_allowed: bool) -> bool:
        if not repeated_locants_are_one_structure(candidate, smiles):
            return False
        opsin_smiles = _validity_gate_name_to_smiles(candidate)
        if (opsin_smiles is None or _shipped_name_round_trip(candidate, smiles) != "verified"
                or not Orthonym._rt_match(smiles, opsin_smiles, omission_allowed)):
            return False
        written = _lone_pair_as_written(smiles)
        return not (_has_lone_pair_stereocentre(written) and not _lone_pair_configuration_verified(candidate, written))

    if reads_back(name, False):
        return name
    if stereo_free_smiles(smiles) is None:         # the input has no stereo: nothing is left out
        return None
    composed = _verified_composed_name(smiles, name)   # the stereo of the input composed on the name
    if composed is not None:
        return composed
    return name if reads_back(name, True) else None
