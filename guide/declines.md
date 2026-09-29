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
    "verified": "unverified"
}
```

The first line is a log message on the error stream: the engine withdrew a candidate that OPSIN could not read. The JSON row is on the normal output.

## The reason codes

| `limit_code` | Label in place of the name | When |
|:--|:--|:--|
| `UNSUPPORTED_ELEMENT` | `inorganic compound (not supported)`, or `<metal> compound (not supported)` | an element outside the ones the rules cover |
| `WILDCARD_ATOMS` | `compound with wildcard atoms (not supported)` | the structure has a `*` atom |
| `STRUCTURE_TOO_LARGE` | `unknown organic compound` | more heavy atoms than the engine's limit |
| `ISOLATED_ATOM` | `unknown organic compound` | a single heavy atom |
| `UNSUPPORTED_RING_SYSTEM` | `unknown organic compound` | a ring system the engine recognises but cannot name correctly yet (with `raise_on_limit=True`) |
| `UNNAMEABLE` | `unknown organic compound` | an organic structure no candidate name passed the checks for |

Most organic declines carry the general `UNNAMEABLE`: no candidate name passed the checks, whatever the reason.

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

`orthonym.classify_limit` returns the reason without raising, and `raise_on_limit=True` turns a decline into an `OrthonymLimitError` ([Python](use/python.md)).

A decline is never silent: a molecule that gets no name gets a label, and `--provenance` gives a reason code. `--batch` prints the label only.
