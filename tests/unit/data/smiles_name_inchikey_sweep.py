"""SMILES-to-name InChIKey sweep over every hardcoded mapping in orthonym.data.

WHY THIS EXISTS
---------------
A recurring, silent defect class in this codebase is *a hardcoded SMILES that does
not match the name attached to it*. Two instances are on record:

* ``data/bicyclo_systems.py`` mapped ``C1CC2CCC1CN2`` to ``"quinuclidine"``. That
  SMILES is 2-azabicyclo[2.2.2]octane (isoquinuclidine, InChIKey
  KPUSZZFAYGWAHZ); quinuclidine is ``C1CN2CCC1CC2`` (SBYHFKPVCBCYGV). Both are
  C7H13N isomers and both keys were already canonical, so nothing flagged it and
  the wrong name shipped.
* The ``cubane`` entry in that same file previously stored a *different* (CH)8
  cage isomer (BOLISNSTKUABPW), so real cubane missed its PIN entirely.

Neither was caught by a unit test, because a test that asserts
``lookup(smiles) == name`` merely restates the data. The only way to catch this
class is to construct the NAMED structure *independently of the table* and compare
constitutions. That is what this module does.

METHOD
------
1. Auto-discover every ``{SMILES: name}`` mapping in ``orthonym.data.**`` by
   walking the package and inspecting module-level dicts. Auto-discovery (rather
   than a hardcoded dict list) is deliberate: a new data table is swept the day it
   lands, with no test edit.
2. For each pair, build the structure the NAME denotes using an independent
   name-to-structure parser, and compare InChIKeys against the stored SMILES.
3. Classify, and split "different compound" from "different stereochemistry",
   because only the former is the bug class above.

AUTHORITY NOTE, BINDING
-----------------------
The independent name-to-structure step is used ONLY as a structure constructor --
to answer "what molecule does this name denote?". It is NEVER evidence about
whether a name is the *preferred* IUPAC name; the Blue Book is the sole authority
for PIN status. A CONSTITUTION_MISMATCH means the table and the name disagree
about the MOLECULE, which is a correctness bug regardless of PIN policy.

It also is not infallible in the other direction: it fails on some legitimate
Blue Book names (verified examples: the lambda-convention name
``N-hydroxy-lambda2-methanamine`` is mis-parsed, and
``2-dithioperoxy-1,3-dithiodicarbonic diamide`` is not parsed at all). Those
surface as NAME_NOT_CONSTRUCTIBLE, which is reported but is NOT a finding.

USAGE (not collected by pytest -- no ``test_`` prefix, needs a JVM)
------------------------------------------------------------------
    .venv/bin/python tests/unit/data/smiles_name_inchikey_sweep.py /tmp/sweep.json

The OPSIN-free regression locks distilled from this sweep live in
``test_smiles_name_consistency.py``, which is what CI runs.
"""

import importlib
import json
import pkgutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

from orthonym.jars import find_jar  # raises JarUnavailable when the jar is missing

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Documented exclusions. Every exclusion is explicit and reasoned -- there are no
# silent skips, because a silent skip is how the isoquinuclidine row survived.
# ---------------------------------------------------------------------------

#: Dicts whose VALUES are substituent, prefix or acyl STEM forms rather than
#: whole-molecule parent names ("methyl", "benzo", "naphtho", "acetyl"). A
#: whole-molecule construction is impossible for these by definition, so they are
#: excluded from the sweep rather than reported as mismatches. They still deserve
#: their own attachment-point-aware verification; that is separate work.
SUBSTITUENT_DICTS = frozenset({
    "orthonym.data.opsin_imports.OPSIN_SUBSTITUENT_NAMES",
    "orthonym.data.opsin_imports.substituent_names_opsin.OPSIN_SUBSTITUENT_NAMES",
    "orthonym.data.fused_heterocycles.FUSED_HETEROCYCLE_PREFIX_STEMS",
    "orthonym.data.opsin_imports.OPSIN_FUSION_COMPONENTS",
    "orthonym.data.opsin_imports.fusion_components_opsin.OPSIN_FUSION_COMPONENTS",
    "orthonym.data.opsin_imports.OPSIN_ACID_STEMS",
    "orthonym.data.opsin_imports.carboxylic_acids_opsin.OPSIN_ACID_STEMS",
    "orthonym.data.opsin_imports.OPSIN_FUNCTIONAL_TERMS",
    "orthonym.data.opsin_imports.functional_terms.OPSIN_FUNCTIONAL_TERMS",
    "orthonym.data.organometallics.LIGAND_NAMES",
    "orthonym.data.organometallics.METALLACYCLE_A_PREFIX",
    "orthonym.data.hw_heteroatoms.HW_PREFIXES",
    "orthonym.data.sugar_names._HALIDE_WORD",
    "orthonym.data.amino_acids.AMINO_ACID_ACYL_NAMES",
    "orthonym.data.amino_acids.AMINO_ACID_ATE_STEMS",
    "orthonym.data.trivial_acids.TRIVIAL_ACID_TO_ACYLATE",
    "orthonym.data.trivial_acids.CHAIN_TO_ACYLATE",
})

#: Bare-element names for monoatomic ions. ``[Na+] -> "sodium"`` is CORRECT usage
#: in the additive/salt sense the Blue Book uses ("sodium chloride", "sodium
#: acetate"), but a name-to-structure parser reasonably builds the NEUTRAL atom
#: from the bare element name, so every such row reports a formula difference of
#: exactly the charge. Carved out by an explicit, checkable rule -- same element,
#: single heavy atom, differing only in charge -- not by name allow-listing.
def _is_bare_element_charge_convention(key_mol, built_mol) -> bool:
    if key_mol is None or built_mol is None:
        return False
    if key_mol.GetNumAtoms() != 1 or built_mol.GetNumAtoms() != 1:
        return False
    a, b = key_mol.GetAtomWithIdx(0), built_mol.GetAtomWithIdx(0)
    return (a.GetSymbol() == b.GetSymbol()
            and a.GetTotalNumHs() == b.GetTotalNumHs() == 0)


NAME_KEYS = ("name", "iupac_name", "retained_name", "preferred_name")


def _extract_name(value):
    """Pull a whole-molecule name out of a dict value of any supported shape."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for nk in NAME_KEYS:
            if isinstance(value.get(nk), str):
                return value[nk]
        names = value.get("names")
        if isinstance(names, list) and names and isinstance(names[0], str):
            return names[0]
    return None


def discover_pairs():
    """Walk ``orthonym.data.**`` and collect every (SMILES, name) mapping.

    Returns ``(pairs, stats)`` where ``pairs`` maps ``(smiles, name)`` to the set
    of ``module.attribute`` origins that declare it.
    """
    import orthonym.data

    pairs = {}
    stats = Counter()
    modules = [m.name for m in pkgutil.walk_packages(
        orthonym.data.__path__, "orthonym.data.")]
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            stats["module_import_failed"] += 1
            continue
        for attr, val in vars(mod).items():
            if attr.startswith("__") or not isinstance(val, dict) or not val:
                continue
            origin = f"{mod_name}.{attr}"
            for key, value in val.items():
                if not isinstance(key, str) or not key or " " in key:
                    continue
                if len(key) > 400:
                    continue
                if Chem.MolFromSmiles(key) is None:
                    continue
                name = _extract_name(value)
                if not name or len(name) > 250:
                    continue
                if origin in SUBSTITUENT_DICTS:
                    stats["excluded_substituent_dict"] += 1
                    continue
                pairs.setdefault((key, name), set()).add(origin)
    stats["modules"] = len(modules)
    stats["swept_pairs"] = len(pairs)
    return pairs, stats


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, None
    try:
        return Chem.MolToInchiKey(mol), mol
    except Exception:
        return None, mol


def build_structures(names):
    """Independently construct each name's structure. One JVM for the batch."""
    proc = subprocess.run(
        ["java", "-jar", find_jar("opsin"), "-o", "smi"],
        input="\n".join(names) + "\n",
        capture_output=True, text=True, timeout=3600,
    )
    lines = proc.stdout.splitlines()
    lines += [""] * (len(names) - len(lines))
    return [ln.strip() or None for ln in lines[:len(names)]]


def classify(key_smiles, built_smiles):
    """Return (verdict, key_inchikey, built_inchikey)."""
    want, key_mol = _inchikey(key_smiles)
    got, built_mol = _inchikey(built_smiles)
    if want is None:
        return "KEY_UNPARSEABLE", want, got
    if got is None:
        return "NAME_NOT_CONSTRUCTIBLE", want, got
    if want == got:
        return "MATCH", want, got
    if _is_bare_element_charge_convention(key_mol, built_mol):
        return "BARE_ELEMENT_CHARGE_CONVENTION", want, got
    if want.split("-")[0] == got.split("-")[0]:
        return "STEREO_MISMATCH", want, got
    return "CONSTITUTION_MISMATCH", want, got


def run(out_path):
    pairs, stats = discover_pairs()
    keys = sorted(pairs)
    built = build_structures([name for _s, name in keys])
    rows = []
    for (smiles, name), b in zip(keys, built):
        verdict, want, got = classify(smiles, b)
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        rows.append({
            "smiles": smiles,
            "name": name,
            "verdict": verdict,
            "key_is_canonical": canonical == smiles,
            "canonical": canonical,
            "key_inchikey": want,
            "name_inchikey": got,
            "built_smiles": b,
            "origins": sorted(pairs[(smiles, name)]),
        })
    counts = Counter(r["verdict"] for r in rows)
    payload = {"stats": dict(stats), "verdicts": dict(counts), "rows": rows}
    Path(out_path).write_text(json.dumps(payload, indent=1))
    for line in (f"{v:6d}  {k}" for k, v in counts.most_common()):
        print(line, file=sys.stderr)
    dead = sum(1 for r in rows if not r["key_is_canonical"])
    print(f"{dead:6d}  NON_CANONICAL_KEY (dead key: lookup is by canonical "
          f"SMILES, so these can never match)", file=sys.stderr)
    return payload


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "sweep.json")
