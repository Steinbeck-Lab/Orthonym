"""An aziridine nitrogen whose configuration RDKit drops when it reads the string, cited in the name.

The nitrogen of an aziridine is a stereocentre: the standard InChI keeps it ('/t4?,6-' for 'CC1[N@@](C1)C(=O)OC'), and
CDK and OPSIN read it. RDKit's SMILES parser removes the tag of such a nitrogen when it carries an acyl or aryl group
(``perception.lone_pair_reading.raw_reading``), so every producer names the molecule without it, and the lone-pair check
of the namer withdraws that name ('methyl 2-methylaziridine-1-carboxylate': no configuration at N-1).
'Trigonal pyramid' (the Blue Book): "The configuration of molecules containing a trigonal pyramidal center
 is described in a similar way to that of tetrahedral centers... by adding a phantom atom (0)", so the
nitrogen is cited 'R' or 'S' at its locant; the nitrogen of an aziridine is position 1,:8222: "the
heteroatom has the locant '1'").

:func:`aziridine_nitrogen_row` runs only when the drawing as written gets no name. It names RDKit's reading (the
drawing without the dropped tags), then adds the descriptor of each such nitrogen to the unit of the name that holds
its 'aziridin' stem: into the stereodescriptor group that opens the unit ('(1R,2S)-'), or as a new group in front of
it, the enclosing marks stepped up when the group lands inside a substituent ('[(1R)-2-methylaziridin-1-yl]',
. A candidate ships only when OPSIN reads it to the full standard InChIKey of the reading that keeps the
tags, and the lone-pair check agrees on the input as written. The label is systematic_verified: the descriptor is
placed by this search, not by a producer certified for the PIN (and never better than the label of the name it was
added to).
"""
import itertools
import re
from typing import List, Optional

from rdkit import Chem

#: the stem of a saturated three-membered ring with one nitrogen, whose nitrogen is position 1 (not 'oxaziridine',
#: 'diaziridine', 'thiaziridine', where the nitrogen can be position 2)
_STEM = re.compile(r"(?<!ox)(?<!di)(?<!hi)(?<!len)aziridin")
#: a stereodescriptor group that opens a unit: '(2S)-', '(2R,3S)-'
_GROUP = re.compile(r"\((?P<items>[0-9]+[RS](?:,[0-9]+[RS])*)\)-")
#: more dropped nitrogens than this are not searched
_MAX_SITES = 2


def _dropped_nitrogens(smiles: str) -> List[int]:
    """Atoms of the reading that keeps the tags (``raw_reading``) that are a nitrogen of a three-membered ring with a
    tag RDKit's own reading of the string does not hold."""
    from ..perception.lone_pair_reading import lone_pair_centres, raw_reading
    raw, clean = raw_reading(smiles), Chem.MolFromSmiles(smiles)
    if raw is None or clean is None or raw.GetNumAtoms() != clean.GetNumAtoms():
        return []
    return [i for i in lone_pair_centres(raw)
            if raw.GetAtomWithIdx(i).GetAtomicNum() == 7 and raw.GetAtomWithIdx(i).IsInRingSize(3)
            and clean.GetAtomWithIdx(i).GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED]


def has_dropped_aziridine_nitrogen(smiles) -> bool:
    """True for a SMILES string that writes an aziridine nitrogen tag RDKit drops, and no other lone-pair centre RDKit
    reads (whose reading this module would not check)."""
    if not isinstance(smiles, str) or '[N@' not in smiles:
        return False
    try:
        from ..perception.lone_pair_reading import lone_pair_centres
        dropped = _dropped_nitrogens(smiles)
        return 0 < len(dropped) <= _MAX_SITES and not lone_pair_centres(Chem.MolFromSmiles(smiles))
    except Exception:  # noqa: BLE001 -- a string the readers refuse is not this module's input
        return False


def _unit_start(name: str, k: int) -> int:
    """The index where the unit of ``name`` that holds the stem at ``k`` begins: after the enclosing mark that is
    open at ``k``, or after the last space before ``k`` outside any mark, or 0."""
    depth = 0
    for i in range(k - 1, -1, -1):
        ch = name[i]
        if ch in ")]}":
            depth += 1
        elif ch in "([{":
            if depth == 0:
                return i + 1
            depth -= 1
        elif ch == " " and depth == 0:
            return i + 1
    return 0


def _with_descriptor(name: str, k: int, label: str) -> Optional[str]:
    """``name`` with '1R' or '1S' (``label``) cited for the aziridine whose stem is at ``k``."""
    from ..rules.isotopes import _escalate_marks_for_descriptor
    u = _unit_start(name, k)
    m = _GROUP.match(name, u)
    if m:
        items = m.group("items").split(",")
        if any(it.startswith("1") and not it[1:2].isdigit() for it in items):
            return None
        merged = ",".join(sorted(items + [f"1{label}"], key=lambda it: int(re.match(r"\d+", it).group())))
        return name[:m.start("items")] + merged + name[m.end("items"):]
    return _escalate_marks_for_descriptor(name[:u] + f"(1{label})-" + name[u:], u)


def _reads_back(name: str, smiles: str) -> bool:
    """OPSIN reads ``name`` to the full standard InChIKey of RDKit's reading of ``smiles`` (every configuration but the
    dropped nitrogens', which RDKit drops alike from OPSIN's structure), and the lone-pair check of the namer, which
    reads the nitrogens as CDK and OPSIN do, agrees on ``smiles`` as written. RDKit's reading that keeps the tags
    (``raw_reading``) is not compared: it reads a nitrogen at a ring-closure digit the other way."""
    from .. import namer as _namer
    from ..perception.molcache import inchikey_of
    opsin_smiles = _namer._validity_gate_name_to_opsin_smiles(name)
    if not opsin_smiles:
        return False
    a, b = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(opsin_smiles)
    if a is None or b is None or inchikey_of(a) != inchikey_of(b) or _dropped_nitrogens(opsin_smiles) == []:
        return False
    return _namer._lone_pair_configuration_verified(name, _namer._lone_pair_as_written(smiles))


def aziridine_nitrogen_row(namer, smiles, written: dict) -> Optional[dict]:
    """The ``name_tiered`` row of ``smiles`` with the configuration of its aziridine nitrogens cited, or None: the
    drawing as written was named, the default tier applies, no such nitrogen, or no candidate reads back."""
    if written.get('tier') != 'abstain' or namer._default_tier_policy_applies():
        return None
    sites = _dropped_nitrogens(smiles)
    if not sites or len(sites) > _MAX_SITES:
        return None
    row = namer.name_tiered(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
    base = row.get('name')
    if row.get('tier') == 'abstain' or not base:
        return None
    stems = [m.start() for m in _STEM.finditer(base)]
    if not stems or len(stems) > 4:
        return None
    found = []
    # one stem per dropped nitrogen, right to left so the earlier indices stay valid; both labels each
    for chosen in itertools.combinations(stems, len(sites)):
        for labels in itertools.product("RS", repeat=len(sites)):
            cand = base
            for k, label in sorted(zip(chosen, labels), reverse=True):
                cand = cand and _with_descriptor(cand, k, label)
            if cand and cand not in found and _reads_back(cand, smiles):
                found.append(cand)
    if len(found) != 1:
        return None         # none, or two spellings of different meaning that both read back: fail closed
    from .decline_fallback import capped
    return capped({**row, 'name': found[0], 'tier': 'systematic_verified', 'is_pin': False,
                   'source': 'aziridine_nitrogen', 'opsin': 'verified', 'verified': 'opsin',
                   'gates_passed': ['self_consistency', 'lone_pair_configuration'],
                   'gate_outcome': 'aziridine_nitrogen_verified'}, row)
