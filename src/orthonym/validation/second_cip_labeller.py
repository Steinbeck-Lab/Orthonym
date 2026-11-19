"""A second CIP labeller for the stereodescriptors OPSIN cannot read.

A name whose stereo layer OPSIN refuses (lowercase ``r``/``s``, "Achiral cyclic compounds"; a ring
system on which OPSIN fails to assign CIP labels) has its constitution checked by OPSIN on the stereo-stripped name
and each descriptor checked against the input's CIP labels at its unit (``validation.stereo_descriptors``), labels
that come from the centres engine (``perception.stereo.assign_stereochemistry``). Where such a name is offered at
the best-effort tier (``assembly.stereo_unread_offer``, user decision 2026-11-19), every unit the name cites must
also carry the same code from RDKit's own implementation of the CIP rules (``rdCIPLabeler``), so two independent
labellers agree on every descriptor that OPSIN does not read. This module is the one place on the naming path that
calls ``rdCIPLabeler`` directly; everywhere else the labels come from ``assign_stereochemistry``.
"""
from typing import Optional

from rdkit import Chem


def cited_codes_agree(name: str, smiles: Optional[str], reduced: str) -> bool:
    """True when ``validation.stereo_descriptors.stereo_block_units`` finds the reading of ``name`` for ``smiles``
    (``reduced`` = ``name`` with its descriptor blocks stripped) and RDKit's ``rdCIPLabeler`` gives every unit the
    reading cites the code the centres engine gives it, and no two descriptor blocks of ``name`` share a locant. False on
    anything it cannot establish."""
    try:
        from rdkit.Chem import rdCIPLabeler
        from ..perception.stereo import assign_stereochemistry
        from ..rules.stereochemistry import strip_stereo_blocks
        from .stereo_descriptors import _cip_descriptor_tokens, stereo_block_units
        # A located token may land on any atom that carries its locant, in any part of the name, so two blocks that
        # share a locant ('(1r,4r)' and '(1s,4s)' on two cyclohexyl groups) can be read swapped and still pass: such a
        # name is not offered (review 2026-11-19).
        blocks = _cip_descriptor_tokens(strip_stereo_blocks(name)[1]) or []
        seen = set()
        for block in blocks:
            locants = {loc for loc, _code in block if loc is not None}
            if locants & seen:
                return False
            seen |= locants
        reading = stereo_block_units(name, smiles, reduced)
        if not reading:
            return False
        units = {u for block in reading for u in block.units}
        if not units:
            return False
        first, second = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(smiles)
        if first is None or second is None:
            return False
        assign_stereochemistry(first)
        rdCIPLabeler.AssignCIPLabels(second)
        for kind, idx in units:
            a = first.GetAtomWithIdx(idx) if kind == 'a' else first.GetBondWithIdx(idx)
            b = second.GetAtomWithIdx(idx) if kind == 'a' else second.GetBondWithIdx(idx)
            if not (a.HasProp('_CIPCode') and b.HasProp('_CIPCode')) or a.GetProp('_CIPCode') != b.GetProp('_CIPCode'):
                return False
        return True
    except Exception:
        return False
