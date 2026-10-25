# Declines

When Orthonym cannot name a molecule, it says so, and what it says is not a name. The plain call returns the empty string and logs the reason, a limit code and a label, as one warning on the `orthonym` logger. The command line prints an empty name, and the reason on the error stream:

```console
$ orthonym "O=[U](=O)=O"

O=[U](=O)=O	UNSUPPORTED_ELEMENT: inorganic compound (not supported)
```

The first line is the (empty) name on the normal output, the second the reason on the error stream. `--on-failure label` (`on_failure="label"` in Python) prints and returns the label in place of a name, such as `inorganic compound (not supported)`, as earlier versions did; `--on-failure raise` (`on_failure="raise"`) stops with an error. The [provenance row](use/provenance.md) marks the row {tier}`abstain` with a reason code in `limit_code`:

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
    "verified": "unverified",
    "spelling_failures": [],
    "producer": null
}
```

The first line is a log message on the error stream: the engine withdrew a candidate that OPSIN could not read. The JSON row is on the normal output. `producer` is `null`: no name ships, so no code is named ([Which code built the name](use/provenance.md#which-code-built-the-name)).

## The reason codes

| `limit_code` | Label (`on_failure="label"`, and the `name` of a provenance row) | When |
|:--|:--|:--|
| `UNSUPPORTED_ELEMENT` | `inorganic compound (not supported)`, or `<metal> compound (not supported)` | an element outside the ones the rules cover, or a compound of a metal that is not admitted to the naming mechanisms and that no general producer names ([Metal compounds](tiers.md#metal-compounds)) |
| `WILDCARD_ATOMS` | `compound with wildcard atoms (not supported)` | the structure has a `*` atom |
| `STRUCTURE_TOO_LARGE` | `unknown organic compound` | an unnamed structure with more than 125 heavy atoms: a label given after the attempt, since the engine sets no size limit and attempts every input |
| `ISOLATED_ATOM` | `unknown organic compound` | a single heavy atom |
| `UNSUPPORTED_RING_SYSTEM` | `unknown organic compound` | a ring system the engine recognises but cannot name correctly yet (with `raise_on_limit=True`) |
| `UNNAMEABLE` | `unknown organic compound` | an organic structure no candidate name passed the checks for |
| `NO_VERIFIED_PIN` | `unknown organic compound` | the default tier built and checked a name, but not a verified PIN and not one of its exceptions; `--emit-tier best-effort` returns that name with its tier |

The codes are the stable part: the keys of `orthonym.errors.LIMIT_CATALOG`. The texts of the labels can change, so match a code, never a label.

Most organic declines carry the general `UNNAMEABLE` (no candidate name passed the checks) or, at the default tier, `NO_VERIFIED_PIN` (a checked name exists, but not the PIN).

Betaine shows the second case. Its strict-path name contains a part the engine records as not the preferred form, so the default tier declines, and a wider tier returns the name labelled {tier}`systematic_verified`:

```console
$ orthonym "C[N+](C)(C)CC(=O)[O-]" --provenance
{"name": "unknown organic compound", "tier": "abstain", "is_pin": false, "source": "abstain", "opsin": "n/a", "gates_passed": [], "gate_outcome": "suppressed", "formula": "C5H11NO2", "limit_code": "NO_VERIFIED_PIN", "stereo_unexpressed": false, "suffix_free_prefix_name": false, "prefix_order_fallback": false, "verified": "unverified", "spelling_failures": [], "producer": null}
$ orthonym "C[N+](C)(C)CC(=O)[O-]" --emit-tier valid
(trimethylazaniumyl)acetate
```

The default-tier row has `"producer": null` although the engine built a name, because nothing ships. With `--provenance` the `valid` row names the code that built it.

Natural-product names are a second case. Where a molecule's preferred name is a bridged fused name that the engine does not build yet, its natural-product name (`diamorphine`, `(9R,13S,14S)-3-methoxy-17-methylmorphinan`) is not the PIN. The default tier declines it with `NO_VERIFIED_PIN`, and a wider tier returns it labelled {tier}`systematic_verified`. `--trivial` also returns the trivial ones, such as `diamorphine`, at the default tier.

At a tier other than `pin`, the provenance row has no label for the name (`null`), and `--on-failure label` prints `(no name — CODE)` for it:

```console
$ orthonym --emit-tier best-effort "O=[U](=O)=O" --on-failure label
(no name — UNSUPPORTED_ELEMENT)
```

## In code

A molecule that gets no name gets the empty string, and the reason is logged:

```pycon
>>> from orthonym import name_compound
>>> name_compound("O=[U](=O)=O")
''
>>> name_compound("O=[U](=O)=O", on_failure="label")
'inorganic compound (not supported)'
```

`orthonym.errors.is_failure_name` is true for the empty string and for a label, and false for a name:

```pycon
>>> from orthonym.errors import is_failure_name
>>> is_failure_name(name_compound("O=[U](=O)=O"))
True
>>> is_failure_name(name_compound("CCO"))
False
```

`orthonym.classify_limit` returns the reason without raising, and `on_failure="raise"` (or `raise_on_limit=True`) turns a decline into an `OrthonymLimitError` ([Python](use/python.md)). For a molecule the default tier declines with `NO_VERIFIED_PIN`, `classify_limit` returns that code and `on_failure="raise"` raises `OrthonymLimitError` with it.

A decline is never silent: a molecule that gets no name gets a logged reason (the limit code and a label), and `--provenance` gives a reason code. `--batch` leaves the name column of that line empty, prints the reason on the error stream, and counts such lines with `--verbose`; with `--batch --provenance` the row of that line holds the reason code.
