# Declines

When Orthonym cannot name a molecule, it says so. The plain call returns a label in place of a name, such as `inorganic compound (not supported)`, and the [provenance row](use/provenance.md) marks the row {tier}`abstain` with a reason code in `limit_code`.

```console
$ orthonym "O=[U](=O)=O" --provenance | python -m json.tool
OPSIN validity gate suppressed unparseable name: 'unknown'
{
    "name": "inorganic compound (not supported)",
    "tier": "abstain",
    "is_pin": false,
    "source": "abstain",
    "opsin": "n/a",
    "gates_passed": [],
    "gate_outcome": "suppressed",
    "formula": "O3U",
    "limit_code": "UNSUPPORTED_ELEMENT",
    "stereo_unexpressed": false,
    "suffix_free_prefix_name": false,
    "prefix_order_fallback": false,
    "verified": "unverified"
}
```

The first line is a log message on the error stream: the engine withdrew a candidate that OPSIN could not read. The JSON row is on the normal output.

## The reason codes

| `limit_code` | Label in place of the name | When |
|:--|:--|:--|
| `UNSUPPORTED_ELEMENT` | `inorganic compound (not supported)`, or `<metal> compound (not supported)` | an element outside the ones the rules cover |
| `WILDCARD_ATOMS` | `compound with wildcard atoms (not supported)` | the structure has a `*` atom |
| `STRUCTURE_TOO_LARGE` | `unknown organic compound` | an unnamed structure with more than 125 heavy atoms: a label given after the attempt, since the engine sets no size limit and attempts every input |
| `ISOLATED_ATOM` | `unknown organic compound` | a single heavy atom |
| `UNSUPPORTED_RING_SYSTEM` | `unknown organic compound` | a ring system the engine recognises but cannot name correctly yet (with `raise_on_limit=True`) |
| `UNNAMEABLE` | `unknown organic compound` | an organic structure no candidate name passed the checks for |
| `NO_VERIFIED_PIN` | `unknown organic compound` | the default tier built and checked a name, but not a verified PIN and not one of its exceptions; `--emit-tier best-effort` returns that name with its tier |

Most organic declines carry the general `UNNAMEABLE` (no candidate name passed the checks) or, at the default tier, `NO_VERIFIED_PIN` (a checked name exists, but not the PIN).

Betaine shows the second case. Its strict-path name contains a part the engine records as not the preferred form, so the default tier declines, and a wider tier returns the name labelled {tier}`systematic_verified`:

```console
$ orthonym "C[N+](C)(C)CC(=O)[O-]" --provenance
{"name": "unknown organic compound", "tier": "abstain", "is_pin": false, "source": "abstain", "opsin": "n/a", "gates_passed": [], "gate_outcome": "suppressed", "formula": "C5H11NO2", "limit_code": "NO_VERIFIED_PIN", "stereo_unexpressed": false, "suffix_free_prefix_name": false, "prefix_order_fallback": false, "verified": "unverified"}
$ orthonym "C[N+](C)(C)CC(=O)[O-]" --emit-tier valid
(trimethylazaniumyl)acetate
```

At a tier other than `pin`, the plain command line prints `(no name — CODE)` instead of the label:

```console
$ orthonym --emit-tier best-effort "O=[U](=O)=O"
(no name — UNSUPPORTED_ELEMENT)
```

## In code

`orthonym.errors.is_failure_name` tells a label from a name:

```pycon
>>> from orthonym import name_compound
>>> from orthonym.errors import is_failure_name
>>> is_failure_name(name_compound("O=[U](=O)=O"))
True
>>> is_failure_name(name_compound("CCO"))
False
```

`orthonym.classify_limit` returns the reason without raising, and `raise_on_limit=True` turns a decline into an `OrthonymLimitError` ([Python](use/python.md)). For a molecule the default tier declines with `NO_VERIFIED_PIN`, `classify_limit` returns that code and `raise_on_limit=True` raises `OrthonymLimitError` with it.

A decline is never silent: a molecule that gets no name gets a label, and `--provenance` gives a reason code. `--batch` prints the label only.
