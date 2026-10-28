"""
Command-line interface for Orthonym.

Usage:
    orthonym "CCO"
    orthonym "CC(=O)O" --style general
    python -m orthonym "SMILES"
"""

import argparse
import functools
import logging
import os
import sys
from typing import List, Optional, Tuple

from . import __version__, name_compound
from . import cli_time_limit as _tl
from .errors import LIMIT_CATALOG, OrthonymLimitError, is_failure_name
from .failure_mode import DEFAULT_MODE as DEFAULT_ON_FAILURE
from .failure_mode import MODES as ON_FAILURE_MODES
from .rdkit_log import rdkit_logs_off, rdkit_logs_on
from .tiers import EMIT_TIERS, default_tier
from .wallclock import WallClockTimeout, clear_leaked_naming_state

_VERSION_HELP = ("Print the version, in the form 'orthonym X.Y.Z', and stop. When --opsin or ORTHONYM_OPSIN "
                 "selects OPSIN's main branch, a second line names the OPSIN build.")


def _opsin_choice_arg(value: str) -> str:
    """``--opsin`` value: 2.9.0, main or main@<commit>."""
    from orthonym.opsin_build import OpsinBuildError, parse_choice
    try:
        parse_choice(value)
    except OpsinBuildError:
        raise argparse.ArgumentTypeError(f"{value!r} is not one of 2.9.0, main or main@<commit>") from None
    return value.strip()


def _add_opsin_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--opsin",
        type=_opsin_choice_arg,
        metavar="{2.9.0,main,main@COMMIT}",
        default=None,
        help=(
            "Which OPSIN reads every name back. 2.9.0 (the default) is the release "
            "jar. main is the newest commit of OPSIN's main branch, which Orthonym "
            "builds with your JDK on first use (about a minute, needs a JDK and "
            "GitHub) and keeps in the jar directory. main@COMMIT is that one "
            "commit of OPSIN's main branch (GitHub is asked once, when it is built, "
            "whether the commit is on that branch), for a repeatable run. The same "
            "as the environment variable "
            "ORTHONYM_OPSIN. OPSIN 2.9.0 stays the default; a jar given by "
            "ORTHONYM_OPSIN_JAR wins over this option. With --fetch-jars, a "
            "main-branch OPSIN is built now instead of at first use."
        ),
    )


def _opsin_option_applied(parsed: argparse.Namespace) -> bool:
    """Make ``--opsin`` the choice of this process, before anything resolves a jar; True when
    ``--version`` was asked for and has been printed."""
    if parsed.opsin is not None:
        os.environ["ORTHONYM_OPSIN"] = parsed.opsin
    if parsed.version:
        print(_version_text())
    return parsed.version


def _jars_ready(parsed: argparse.Namespace) -> Optional[int]:
    """Fetch the jars (``--fetch-jars``, which also builds a main-branch OPSIN that was asked
    for) or check that they are there. The exit status when the run ends here, else None."""
    from orthonym.jars import (
        JarDirUnsafe,
        JarUnavailable,
        OpsinChoiceError,
        fetch_all,
        fetch_opsin_choice,
        require_all,
    )
    if parsed.fetch_jars:
        try:
            fetch_all(verbose=True)
            fetch_opsin_choice(verbose=True)
        except JarUnavailable as exc:
            print(f"orthonym: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        require_all()
    except JarUnavailable as exc:
        # a mistyped choice, and a jar directory that other users can write, are not met by reduced mode
        hint = "" if isinstance(exc, (OpsinChoiceError, JarDirUnsafe)) else (
            "\northonym: to name without the jars (no OPSIN check, RDKit stereo labels), "
            "set ORTHONYM_ALLOW_REDUCED=1")
        print(f"orthonym: {exc}{hint}", file=sys.stderr)
        return 2
    return None


def _version_text() -> str:
    """``orthonym X.Y.Z``; with a main-branch OPSIN selected, a second line says which build.

    Offline and cheap: it looks for the build in the jar directory and never builds or downloads."""
    text = f"orthonym {__version__}"
    from orthonym import jars
    try:
        if jars.opsin_choice() is None:
            return text
    except jars.JarUnavailable as exc:
        return f"{text}\nOPSIN: {exc}"
    try:
        info = jars.active_opsin(download=False)
    except jars.JarUnavailable:
        return f"{text}\nOPSIN {os.environ.get('ORTHONYM_OPSIN', '').strip()} (not built yet; " \
               f"`orthonym --fetch-jars` builds it)"
    commit = f" {info['commit'][:12]}" if info["commit"] else ""
    return f"{text}\nOPSIN {info['label']}{commit}"


def _emit_tier_flags(emit_tier: str) -> dict:
    """Map a ``--emit-tier`` value to the Orthonym general-engine flag set (``orthonym.tiers.emit_tier_flags``,
    the one home of the invariant table)."""
    from .tiers import emit_tier_flags
    return emit_tier_flags(emit_tier)


def _engine_only(parsed) -> int:
    """``--engine-only``: the general engine's own name, as JSON, or that it declined.

    A diagnostic, not a public naming entry: it perceives and classifies the string as written, with
    no input scope (``namer._lone_pair_input_enter``), so it does not name the canonical spelling a
    public call names. Left so on purpose: it shows the engine on the structure as it was drawn."""
    import json as _json

    from rdkit import Chem as _Chem

    from orthonym.assembly.general_engine import name_general
    from orthonym.namer import Orthonym
    from orthonym.validation.e1_certificate import verify_certificate
    namer = Orthonym(style=parsed.style,
                      _disable_opsin_validity_gate=True)
    mol = _Chem.MolFromSmiles(parsed.smiles)
    if mol is None:
        print("Error: invalid SMILES", file=sys.stderr)
        return 1
    with _tl.limit(parsed.time_limit):
        feats = namer._perceive(
            mol, parsed.smiles, _Chem.MolToSmiles(mol, canonical=True))
        namer._classify(feats)
        res = name_general(mol, feats)
    if res is None:
        print(_json.dumps({
            "engine": "refused",
            "note": "fail-closed; see log for the refusal reason",
        }))
        return 0
    verdict = verify_certificate(mol, res)
    print(_json.dumps({
        "engine": res.name,
        "e1_ok": verdict.ok,
        "e1_reason": verdict.reason,
        "warning": "diagnostic output — E1-audited but NOT "
                   "OPSIN-RT-gated; never a PIN",
    }))
    return 0


def _unnamed_line(smiles: str, code: str, label: str) -> str:
    """The line on the error stream for a structure that got no name."""
    return f"{smiles}\t{code}: {label}"


#: The code and label of an input RDKit cannot read. It is an input error, not a limit of the
#: engine, so it is not in ``LIMIT_CATALOG``: the Python API raises ``ValueError`` for it.
_UNREADABLE = ("INVALID_SMILES", "RDKit cannot read this SMILES")


def _readable(smiles: str) -> bool:
    """Whether RDKit reads the user's SMILES. When it does not, RDKit says why on the error
    stream (the one RDKit message the command line shows: it is about the user's input) and
    the reason line follows."""
    from rdkit import Chem
    with rdkit_logs_on("rdApp.error"):
        if Chem.MolFromSmiles(smiles) is not None:
            return True
    print(_unnamed_line(smiles, *_UNREADABLE), file=sys.stderr)
    return False


class _UnnamedReporter(logging.Filter):
    """Print, on the error stream, the line for each structure the library reports as unnamed.

    ``failure_mode`` logs one warning per unnamed structure on the logger ``orthonym``, with the
    SMILES, the limit code and the label as record attributes. This filter turns it into the
    line of the command line and drops the record, so the reason is not printed twice."""

    def filter(self, record: logging.LogRecord) -> bool:
        code = getattr(record, "orthonym_code", None)
        if code is None:
            return True
        print(_unnamed_line(record.orthonym_smiles, code, record.orthonym_label),
              file=sys.stderr)
        return False


def _reports_unnamed(fn):
    """Run ``fn`` (``main``) with the unnamed-structure line printed for each failure and
    RDKit's own warnings and errors off: the engine's internal SMILES probes make them, and
    only the user's own input is checked with the errors on (:func:`_readable`)."""
    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        reporter = _UnnamedReporter()
        logger = logging.getLogger("orthonym")
        logger.addFilter(reporter)
        try:
            with rdkit_logs_off("rdApp.warning", "rdApp.error"):
                return fn(*args, **kwargs)
        finally:
            logger.removeFilter(reporter)
    return _wrapper


def _add_on_failure_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--on-failure",
        dest="on_failure",
        choices=list(ON_FAILURE_MODES),
        default=DEFAULT_ON_FAILURE,
        help=("What to give for a structure that gets no name. empty (the default): an empty "
              "name, and on the error stream a line with the SMILES, the limit code and the "
              "reason, '<SMILES><tab><CODE>: <reason>'. label: the reason as the name, "
              "'unknown organic compound' for example. raise: stop with an error (status 1; "
              "with --batch the line gets ERROR: and the run ends with status 1). It applies "
              "to a name, --confidence and --batch; --provenance (with --batch too), "
              "--dump-tree and --engine-only report the structure as it is. With --batch "
              "--verbose the count of lines without a name is printed.")
    )


def _wants_tier_row(emit_tier: str, provenance: bool) -> bool:
    """Whether a run names through ``Orthonym.name_tiered`` (a tier row) and not ``name_compound``.

    One question for one SMILES and for ``--batch``: a tier other than ``pin``, or ``--provenance``."""
    return emit_tier != "pin" or provenance


def _tiered_namer(style: str, emit_tier: str, *, enable_triviality_controller: bool = False,
                  trivial_fallback: bool = False, binding_proof: str = "off"):
    """The ``Orthonym`` that names the rows of ``--emit-tier``, ``--provenance`` and
    ``--binding-proof``. One home for one SMILES and for ``--batch``, which builds it once and
    names every line with it."""
    from orthonym.namer import Orthonym
    tier_flags = _emit_tier_flags(emit_tier)
    return Orthonym(
        style=style,
        enable_triviality_controller=enable_triviality_controller,
        trivial_fallback=trivial_fallback,
        general_fallback=tier_flags["general_fallback"],
        general_fallback_unverified=tier_flags["general_fallback_unverified"],
        allow_aromatic_general=tier_flags["allow_aromatic_general"],
        full_coverage=tier_flags.get("full_coverage", False),
        binding_proof=binding_proof,
    )


def _provenance_json(row: dict) -> str:
    """The row of ``--provenance``, as the one line of JSON that both paths print."""
    import json
    return json.dumps(row)


def _tiered_row_is_named(row: dict) -> bool:
    """Whether a tier row has a name: a clean abstain has none (``name_tiered``, T5)."""
    return bool(row["name"]) and row["tier"] != "abstain"


def _decline_label(row: dict, smiles: str, code: str, default: bool) -> str:
    """The label of a structure with no name: for the default the label ``name_compound`` gives it
    ('platinum compound (not supported)', 'unknown organic compound'), as the default gave it before it was
    the ``best-effort`` tier; for a tier named with ``--emit-tier`` the row's label or the catalogued message of
    the code."""
    if default:
        from .namer import _descriptive_fallback
        return _descriptive_fallback(smiles)
    return row["name"] or LIMIT_CATALOG.get(code, {}).get("message", "")


def _tiered_name_text(row: dict, smiles: str, mode: str,
                      default: bool = False) -> Tuple[str, Optional[str]]:
    """The bare name of a tiered row (``--emit-tier``) or, for a structure with no name, what
    ``--on-failure`` says: with ``label`` the row's label or '(no name - <code>)' as always, and for the
    default (``default``: no ``--emit-tier`` given) the label and the reason line of ``name_compound``
    (:func:`_decline_label`).

    Returns the text for the name column and the reason line for the error stream (None when
    there is nothing to say); ``raise`` raises the limit. One decision for one SMILES and for
    ``--batch``."""
    if mode == "label":
        if row["name"]:
            return row["name"], None
        code = row["limit_code"] or row["tier"]
        return (_decline_label(row, smiles, code, True) if default else f"(no name — {code})"), None
    if _tiered_row_is_named(row):
        return row["name"], None
    code = row["limit_code"] or "UNNAMEABLE"
    label = _decline_label(row, smiles, code, default)
    if mode == "raise":
        raise OrthonymLimitError(code, label, smiles=smiles)
    return "", _unnamed_line(smiles, code, label)


def _timeout_text(smiles: str, seconds: float, mode: str, provenance: bool) -> Tuple[str, Optional[str]]:
    """What a structure that ran out of ``--time-limit`` gets, decided as:func:`_tiered_name_text` decides
    for a structure with no name: ``--provenance`` its row, ``label`` the label, ``raise`` raises the limit
    (code ``TIME_LIMIT``), ``empty`` an empty name and the reason line for the error stream."""
    if provenance:
        return _provenance_json(_tl.row(smiles, seconds)), None
    if mode == "label":
        return _tl.label(seconds), None
    if mode == "raise":
        raise _tl.error(smiles, seconds)
    return "", _unnamed_line(smiles, _tl.CODE, _tl.label(seconds))


def _single_timed_out(parsed: argparse.Namespace) -> int:
    """The end of a one-SMILES run whose naming ran out of ``--time-limit``: the same output as for a
    structure that gets no name, and the status of that case (``raise`` exits 1)."""
    clear_leaked_naming_state()     # ``wall_clock_limit`` put it back already; empty unless the limit has a bug
    if parsed.engine_only:          # the JSON of its other declines (the structure as it is, whatever --on-failure says)
        import json
        print(json.dumps({"engine": "refused", "limit_code": _tl.CODE, "note": _tl.label(parsed.time_limit)}))
        return 0
    try:
        text, reason = _timeout_text(parsed.smiles, parsed.time_limit, parsed.on_failure, parsed.provenance)
    except OrthonymLimitError as e:
        print(_unnamed_line(parsed.smiles, e.code, e.message), file=sys.stderr)
        return 1
    if parsed.verbose and not parsed.provenance:
        print(f"SMILES: {parsed.smiles}\nStyle:  {parsed.style}\nName:   {text}")
    else:
        print(text)
    if reason is not None:
        print(reason, file=sys.stderr)
    return 0


#: The one note of a run on the error stream: printed only when the default (no ``--emit-tier``) returned a name
#: below ``pin_verified`` (a name that is not certified as the preferred name; its tier says how it was checked).
_BELOW_PIN_NOTE = ("orthonym: note: the default returns the PIN when it can, else a systematic name that is not "
                   "the PIN; --provenance shows the tier and the check of each name, --emit-tier pin gives PIN only")


def _is_below_pin(tier: Optional[str]) -> bool:
    """Whether a returned name earned a tier below ``pin_verified`` (None: no name)."""
    return tier is not None and tier not in ("pin_verified", "abstain")


_CONFIDENCE_NEEDS_PIN = ("--confidence cannot be combined with --provenance: a tier row has no coverage "
                         "score")

#: The flags that look inside the naming of ONE structure (``--format`` only sets the layout of
#: ``--dump-tree``): a file of SMILES has no such single run to show, and ``--batch`` used to ignore them.
_ONE_SMILES_ONLY = (("validation_stats", "--validation-stats"), ("dispatch_stats", "--dispatch-stats"),
                    ("engine_only", "--engine-only"), ("dump_tree", "--dump-tree"), ("format", "--format"))


def _check_flag_combinations(parser: argparse.ArgumentParser, parsed: argparse.Namespace) -> None:
    """Usage error (status 2) for flags that cannot be asked for together, before anything is read
    or any jar is checked."""
    # A tier row has no coverage score, so the two cannot be asked for together (one SMILES
    # used to drop --confidence without a word).
    if parsed.confidence and parsed.provenance:
        parser.error(_CONFIDENCE_NEEDS_PIN)
    one_only = [flag for dest, flag in _ONE_SMILES_ONLY if getattr(parsed, dest)] if parsed.batch else []
    if one_only:
        parser.error(f"{', '.join(one_only)} cannot be combined with --batch "
                     "(for one SMILES, not for a file of them)")


@_reports_unnamed
def main(args: List[str] = None) -> int:
    """Main entry point for CLI."""
    # Before anything can start the JVM: a native crash must print the Python
    # stack, not only a JVM hs_err file (orthonym.diagnostics, audit 2026-09-03).
    from orthonym.diagnostics import enable_crash_traceback
    enable_crash_traceback()
    parser = argparse.ArgumentParser(
        description=("Orthonym: IUPAC names for chemical structures. Give a SMILES string "
                     "and get its IUPAC name, checked by reading it back with OPSIN. The default "
                     "returns the preferred IUPAC name (PIN) when the strict path builds one, and "
                     "otherwise a systematic name that is not the PIN (--emit-tier pin gives PINs "
                     "only). A structure that gets no name has an empty name and a reason on the "
                     "error stream (--on-failure). --provenance shows how each name was built and "
                     "checked."),
        prog="orthonym"
    )

    parser.add_argument(
        "smiles",
        nargs="?",
        help="The structure, as a SMILES string (in quotes)."
    )

    parser.add_argument(
        "--style",
        choices=["pin", "general", "cas"],
        default="pin",
        help=("Naming style: pin (aim at the Preferred IUPAC Name; the default), "
              "general or cas. general allows general IUPAC forms where the "
              "recommendations offer one, for example functional class names "
              "('dimethyl sulfoxide', 'methyl isocyanate'), hydrate names ('oxalic "
              "acid dihydrate') and the axial descriptors Ra and Sa. The PIN-only "
              "rule (--emit-tier pin) applies to pin only, so with general or cas a "
              "name that pin declines with NO_VERIFIED_PIN is returned; the default "
              "tier of general and cas is not widened. cas is "
              "accepted and at present gives the names of general, except that "
              "hydrate names and axial descriptors keep their pin form.")
    )

    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help=_VERSION_HELP
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help=("Also print the SMILES and the style. With --batch, also say how many lines "
              "were read, how many failed and how many got no name.")
    )

    parser.add_argument(
        "--batch", "-b",
        type=str,
        help=("Name every SMILES in this file, one per line, and print one "
              "'SMILES<tab>name' line each; the name is empty for a line that gets no name "
              "(--on-failure). It names each line as one SMILES is named: the default, or "
              "the names of the --emit-tier given, and --provenance "
              "prints the JSON row of each line in place of the name. "
              "--validation-stats, --dispatch-stats, --engine-only, --dump-tree and --format "
              "look inside one SMILES and stop with a usage error here.")
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="With --batch: write the lines to this file instead of the screen."
    )

    parser.add_argument(
        "--confidence",
        action="store_true",
        help=("Print the name with a coverage score, the tier the name earned (for a "
              "tier above pin), the part of the engine that built it, and the parts of "
              "the score. Not with --provenance.")
    )

    # a phase Plan-03 Task 4 (internal notes telemetry):
    # Print the OPSIN-grammar pre-validation seven-bucket counter
    # histogram to stderr after naming. Always goes through the
    # `Orthonym(...).get_validation_stats` accessor — never reads
    # a module-global counter .
    parser.add_argument(
        "--validation-stats",
        action="store_true",
        help=("After the name, print on the error stream how often the OPSIN grammar "
              "pre-check passed or repaired a candidate name. Not with --batch.")
    )

    # a phase Plan-03 Task 9 (internal notes telemetry):
    # Print the CFR class-first dispatch counter histogram to stderr
    # after naming. Default-OFF (additive; existing CLI behavior
    # unchanged). Goes through Orthonym(...).get_dispatch_stats
    # accessor per internal notes + (never reads a module-global
    # counter). Stderr-only emission keeps stdout clean for piped use.
    parser.add_argument(
        "--dispatch-stats",
        action="store_true",
        help=("After the name, print on the error stream which compound-class routes "
              "the engine took. Not with --batch.")
    )

    # a phase Plan-04 (internal notes) + a phase: --dump-tree emits the
    # NameTreeNode IR for the given SMILES. Default format is "text"
    # (chemist-readable indented tree); --format json emits dataclasses.asdict
    # JSON for machine consumption. The CLI invokes Orthonym.name_with_tree(smi)
    # which returns NamingResult(name, tree, atom_to_locant_hint). a phase:
    # every reachable handler plus the ion/retained boundary fallback populates
    # a tree, so the dump always shows a structured-or-coarse IR (recursive).
    parser.add_argument(
        "--dump-tree",
        action="store_true",
        help="Print the parts of the name as a tree instead of the name. Not with --batch."
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default=None,       # None, not "text": --batch can tell that it was given
        help="Layout for --dump-tree: text (an indented tree; the default) or json. Not with --batch."
    )

    # a phase Triviality Controller (/02/03 + internal notes).
    triv_group = parser.add_argument_group("Retained parent names")
    triv_group.add_argument(
        "--enable-triviality-controller",
        dest="enable_triviality_controller",
        action="store_true",
        default=False,
        help=(
            "Use a retained parent name (benzene, phenol, aniline, benzoic acid and"
            " others) where the IUPAC 2013 recommendations prefer it to the "
            "systematic one (P-15.1.8). Every change is checked by an OPSIN round "
            "trip. Off by default; ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1 turns it"
            " on too."
        ),
    )

    # Task 1.9 (PIN-policy): --trivial fallback flag. Default OFF (PIN fails
    # closed). When set, a general-only (PIN-denied) retained name is returned
    # ONLY when the default pipeline could not derive a PIN — a fallback, never
    # a downgrade of a derivable PIN.
    parser.add_argument(
        "--trivial",
        dest="trivial_fallback",
        action="store_true",
        default=False,
        help=(
            "When no preferred name can be built, also allow a retained trivial "
            "name that is not a preferred name. A preferred name that can be built "
            "is never replaced (glycerol stays propane-1,2,3-triol). Without this "
            "option, two kinds of retained trivial name are still used at the wider"
            " tiers, and the PIN-only tier declines them: names from a small "
            "last-resort table, and the trivial natural-product names of molecules "
            "whose preferred bridged fused name the engine does not build yet "
            "(diamorphine). --provenance labels both systematic_verified, source "
            "trivial_retained. A natural-product name built on a parent, such as "
            "'(9R,13S,14S)-3-methoxy-17-methylmorphinan', is not a trivial name, "
            "and --trivial does not return it at the PIN-only tier."
        ),
    )

    #: confidence-tiered output surface. Default 'pin' is the
    # existing PIN-or-abstain behavior byte-identically; 'valid' adds
    # RT-verified general-engine names (T3); 'best-effort' additionally
    # ships E1-certified names OPSIN could not verify (T4). No tier ever
    # ships a name OPSIN parsed to a DIFFERENT structure.
    parser.add_argument(
        "--emit-tier",
        dest="emit_tier",
        choices=list(EMIT_TIERS),
        default=None,
        help=(
            "Which names to return. Without this option: the best-effort tier (the preferred "
            "IUPAC name (PIN) when the strict PIN path builds one, otherwise a systematic name "
            "that is not the PIN; the run says so once on the error stream, and --provenance "
            "shows the tier of each name), and the pin tier for --style general and cas and "
            "when no OPSIN reads the names back. pin: a name only when the pipeline can "
            "build the preferred IUPAC name (PIN), that is when the strict PIN path built "
            "the name and verified it (tier pin_verified); this was the default up to "
            "release 1.0.x. The exceptions are names from "
            "the natural-product and metal-complex lists, the name formats absent from "
            "OPSIN's grammar (inositols; phanes, checked by building their structure), PIN "
            "spellings that OPSIN reads back through a twin spelling of the same structure "
            "(ethan(2H)ol, diphenyliodanuide, methane-SO-thioperoxol) or through their acids (dianhydrides), PINs whose "
            "stereodescriptors OPSIN cannot read, for which the PIN-only tier compares the "
            "constitution, and, with --trivial, a retained trivial name. Otherwise it "
            "declines, with the reason code NO_VERIFIED_PIN when it built a name that is "
            "not a verified PIN. The rule applies to --style pin. "
            "valid: also names from the general engine. complete: also general "
            "names for aromatic and heterocyclic ring systems. best-effort: also "
            "the last-resort producers, von Baeyer and spiro names for ring systems "
            "of up to 100 skeletal atoms and 15 rings (the other tiers build them up to "
            "40 atoms and 8 rings), and adducts with a one-atom ion, and, for a molecule that has no other name, the name the PIN-only tier ships although OPSIN cannot read its stereodescriptors, with the label of the PIN-only tier (pin_unverified). full-coverage: also the coordination-name "
            "builder for metal tetrapyrrole and corrin complexes, which builds a "
            "name or declines. The tiers above pin also return the names the PIN-only "
            "tier declines, each labelled with its tier; there every name must "
            "pass a full-InChIKey OPSIN round trip, except a name from the "
            "natural-product and metal-complex lists, a twin-checked name for which the engine has no other spelling (a halogen -uide anion, the anion of a thioperoxol, an organometallic name with halido ligands) and the best-effort name described above, so the other name formats absent "
            "from OPSIN's grammar are declined there. --provenance shows each "
            "name's tier."
        ),
    )
    parser.add_argument(
        "--provenance",
        action="store_true",
        default=False,
        help=(
            "Print a JSON row instead of the bare name, with the keys name, tier, "
            "is_pin, source, opsin, gates_passed, gate_outcome, formula, "
            "limit_code, stereo_unexpressed, suffix_free_prefix_name, "
            "prefix_order_fallback, verified, spelling_failures and producer. "
            "'verified' is opsin (OPSIN read the whole name back to the same molecule), "
            "opsin_constitution (the same constitution; the stereodescriptors were not confirmed by "
            "OPSIN), identity (a name from an exact-match list: a metal-complex name found by the "
            "input's exact InChIKey, or a natural-product parent name found by its exact structure; "
            "OPSIN cannot read these names), opsin_split (read back to the input without its dative "
            "bonds, which the drawing as given has no key for), opsin_twin (a PIN spelling OPSIN has "
            "no grammar for, read back through a twin spelling of the same structure), opsin_acids (a poly-anhydride, read through its acids), reconstructor (a name with a count word above 50, which OPSIN cannot read: the structure built from its text) or unverified "
            "(no read-back recorded). 'gate_outcome' says what the final OPSIN check did, for example "
            "self_consistency_verified, suppressed, not_run, carveout:<class>, twin_verified:<rules>, acid_split_verified:<rule> or reconstructed:<rule>. "
            "'spelling_failures' lists the spelling rules of the PIN (rule id and detail) that a "
            "name in PIN form breaks, and the rule of a recorded non-PIN part that lowered the "
            "name; such a name is not labelled pin_verified, and pin declines it. With --batch, "
            "each line is the SMILES, a tab and the row, on one line. Not with --confidence."
        ),
    )
    parser.add_argument(
        "--engine-only",
        dest="engine_only",
        action="store_true",
        default=False,
        help=(
            "For inspection only: skip the strict path and print, as JSON, the "
            "general engine's own name (checked for atom coverage, not by OPSIN) or"
            " that it declined. Never a preferred name; do not use it as a name. "
            "Not with --batch."
        ),
    )

    parser.add_argument(
        "--fetch-jars",
        action="store_true",
        help=(
            "Download the OPSIN and centres jars if they are missing, check each "
            "against the SHA-256 checksum recorded in Orthonym, print where they "
            "are, and stop (exit status 1 on failure). A jar given by "
            "ORTHONYM_OPSIN_JAR or ORTHONYM_CENTRES_JAR is used as given, without "
            "the checksum check."
        ),
    )
    _add_opsin_option(parser)
    parser.add_argument(
        "--binding-proof",
        dest="binding_proof",
        choices=["off", "audit", "enforce"],
        default="off",
        help=(
            "An extra check that every part of a general-engine name still maps "
            "onto its atoms in the final name: off (the default), audit (record the"
            " result; the name never changes), or enforce (also decline when the "
            "check fails). Works for one SMILES and for --batch."
        ),
    )
    _add_on_failure_option(parser)
    _tl.add_option(parser)

    parsed = parser.parse_args(args)

    if _opsin_option_applied(parsed):
        return 0

    _check_flag_combinations(parser, parsed)

    # Nothing to name: the help needs no jars, so print it before the jar check (which
    # downloads a missing jar and makes a first call without arguments slow).
    if not parsed.smiles and not parsed.batch and not parsed.fetch_jars:
        parser.print_help()
        return 1

    status = _jars_ready(parsed)
    if status is not None:
        return status

    # the tier: the one asked for, else the default (best-effort when OPSIN reads the names back)
    emit_tier_given = parsed.emit_tier is not None
    parsed.emit_tier = parsed.emit_tier or default_tier(parsed.style)
    parsed.time_limit = _tl.prepare(parsed.time_limit, parsed.style, parsed.emit_tier)

    # Batch processing mode
    if parsed.batch:
        return _process_batch(parsed.batch, parsed.output, parsed.style,
                              parsed.verbose, parsed.confidence,
                              getattr(parsed, 'trivial_fallback', False),
                              getattr(parsed, 'binding_proof', 'off'),
                              parsed.on_failure,
                              emit_tier=parsed.emit_tier,
                              emit_tier_given=emit_tier_given,
                              provenance=parsed.provenance,
                              enable_triviality_controller=getattr(
                                  parsed, 'enable_triviality_controller', False),
                              time_limit=parsed.time_limit)

    # Single SMILES mode (parsed.smiles is set: without it and without --batch the help check
    # above returned)
    if not _readable(parsed.smiles):
        return 1

    # a call that outlasts the alarm ends the run (cli_time_limit); --on-failure raise exits 1, as _single_timed_out does
    _tl.on_hard_stop(lambda: _single_timed_out(parsed), int(parsed.on_failure == "raise" and not parsed.provenance))
    try:
        # a phase Plan-03 Task 4: --validation-stats branch goes
        # through Orthonym(...).get_validation_stats per
        # (never read a module-global counter). Branch is taken BEFORE
        # the existing default `print(name)` so the histogram is the
        # only stderr output path when the flag is on.
        if parsed.validation_stats:
            from orthonym.namer import Orthonym
            namer = Orthonym(style=parsed.style, **_emit_tier_flags(parsed.emit_tier))
            name = _tl.timed(parsed.time_limit, namer.name, parsed.smiles, on_failure=parsed.on_failure)
            print(name)
            stats = namer.get_validation_stats()
            print(f"Validation stats: {stats}", file=sys.stderr)
            return 0

        # a phase Plan-03 Task 9 + a phase: --dispatch-stats
        # branch goes through Orthonym(...).get_dispatch_stats AND
        # Orthonym(...).get_inner_dispatch_stats per internal notes +
        # + (never read a module-global counter). Default-OFF;
        # stderr-only. Histogram is sorted by count descending for
        # readability; StoutClass members print by.name (uppercase
        # identifier). Inner handler_ids print as-is (lowercase snake_case).
        if parsed.dispatch_stats:
            from orthonym.namer import Orthonym
            from orthonym.routing.dispatch_table import StoutClass
            namer = Orthonym(style=parsed.style, **_emit_tier_flags(parsed.emit_tier))
            name = _tl.timed(parsed.time_limit, namer.name, parsed.smiles, on_failure=parsed.on_failure)
            print(name)
            stats = namer.get_dispatch_stats()
            print("\n--- CFR Dispatch Stats (Phase 158 outer) ---", file=sys.stderr)
            # Sort by count descending; tie-break by class_id.value for determinism
            for class_id, count in sorted(
                stats.items(),
                key=lambda kv: (-kv[1], getattr(kv[0], "value", str(kv[0]))),
            ):
                class_name = (
                    class_id.name if isinstance(class_id, StoutClass) else str(class_id)
                )
                print(f"  {class_name}: {count}", file=sys.stderr)
            # a phase: also print inner-dispatch counters.
            inner_stats = namer.get_inner_dispatch_stats()
            print("\n--- Inner Dispatch Stats (Phase 160 inner) ---", file=sys.stderr)
            if not inner_stats:
                print("  (no inner-dispatch hits; molecule routed via CFR fast-path)",
                      file=sys.stderr)
            else:
                for handler_id, count in sorted(
                    inner_stats.items(),
                    key=lambda kv: (-kv[1], kv[0]),
                ):
                    print(f"  {handler_id}: {count}", file=sys.stderr)
            return 0

        # a phase Plan-04 (internal notes): --dump-tree dispatches BEFORE
        # the default print(name) branch so the IR is the sole stdout
        # emission. Default format is "text" (chemist-readable indented
        # tree); --format json emits dataclasses.asdict JSON.
        if parsed.dump_tree:
            from orthonym.namer import Orthonym
            namer = Orthonym(style=parsed.style, **_emit_tier_flags(parsed.emit_tier))
            result = _tl.timed(parsed.time_limit, namer.name_with_tree, parsed.smiles)
            if parsed.format == "json":
                import dataclasses
                import json
                if result.tree is not None:
                    tree_dict = dataclasses.asdict(result.tree)
                else:
                    tree_dict = None
                payload = {
                    "name": result.name,
                    "tree": tree_dict,
                    "atom_to_locant_hint": result.atom_to_locant_hint,
                }
                print(json.dumps(payload, indent=2, default=str))
            else:
                _print_tree_text(result)
            return 0

        # a phase: thread the triviality-controller flag through
        # name_compound. getattr defensive pattern preserves backwards compat.
        # Task 1.9: also thread the --trivial fallback flag.
        name_kwargs = {
            "enable_triviality_controller": getattr(
                parsed, "enable_triviality_controller", False),
            "trivial_fallback": getattr(parsed, "trivial_fallback", False),
            "on_failure": parsed.on_failure,
        }

        #: --engine-only diagnostic branch (before the tier branch: a
        # pure inspection surface, never a production emit path).
        if getattr(parsed, "engine_only", False):
            return _engine_only(parsed)

        # Tiered-output branch: a tier above pin (the default is the best-effort tier, whose names below
        # pin_verified earn the one note on the error stream) and --provenance. Constructs the namer directly
        # (the tier flags are namer-level), prints JSON with --provenance or the bare name otherwise.
        # --emit-tier pin without --provenance stays on the name_compound path below, which is byte-identical
        # to the default of releases up to 1.0.x.
        _emit_tier = getattr(parsed, "emit_tier", "pin")
        #: --binding-proof also selects this branch, because it is the
        # only single-SMILES surface that constructs the namer directly (the
        # default path goes through name_compound). Without it the flag would
        # parse and then silently do nothing -- the same failure mode the
        # ValueError in Orthonym.__init__ exists to prevent. With --confidence it stays on
        # name_compound, as in --batch: a tier row has no coverage score to print.
        _binding_proof = getattr(parsed, "binding_proof", "off")
        if (_wants_tier_row(_emit_tier, parsed.provenance) and not parsed.confidence
                or (_binding_proof != "off" and not parsed.confidence)):
            return _print_tier_row(parsed, name_kwargs, _emit_tier, emit_tier_given, _binding_proof)

        name_kwargs["emit_tier"] = _emit_tier
        if parsed.confidence:
            result = _tl.timed(parsed.time_limit, name_compound, parsed.smiles, style=parsed.style,
                               include_confidence=True,
                               binding_proof=_binding_proof, **name_kwargs)
            _print_confidence(result)
            if not emit_tier_given and _is_below_pin(result.get('tier')):
                print(_BELOW_PIN_NOTE, file=sys.stderr)
        elif parsed.verbose:
            name = _tl.timed(parsed.time_limit, name_compound, parsed.smiles, style=parsed.style, **name_kwargs)
            print(f"SMILES: {parsed.smiles}")
            print(f"Style:  {parsed.style}")
            print(f"Name:   {name}")
        else:
            name = _tl.timed(parsed.time_limit, name_compound, parsed.smiles, style=parsed.style, **name_kwargs)
            print(name)

        return 0

    except WallClockTimeout:
        return _single_timed_out(parsed)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except OrthonymLimitError as e:
        # --on-failure raise: the same line as for an unnamed structure, and the status of an error
        print(_unnamed_line(parsed.smiles, e.code, e.message), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1
    finally:
        _tl.on_hard_stop(None)


def _print_tier_row(parsed, name_kwargs: dict, emit_tier: str, emit_tier_given: bool, binding_proof: str) -> int:
    """One SMILES through ``Orthonym.name_tiered``: a tier above pin and ``--provenance``. Prints the JSON row
    with ``--provenance``, else the bare name; without ``--emit-tier`` a name below pin_verified earns the one
    note on the error stream, and a structure with no name the label of ``name_compound``."""
    namer = _tiered_namer(
        parsed.style, emit_tier,
        enable_triviality_controller=name_kwargs["enable_triviality_controller"],
        trivial_fallback=name_kwargs["trivial_fallback"],
        binding_proof=binding_proof,
    )
    row = _tl.timed(parsed.time_limit, namer.name_tiered, parsed.smiles)
    if parsed.provenance:
        print(_provenance_json(row))
        return 0
    # Composer1 Task 5 fix: a clean abstain has name=None by design (honest "no name" contract, see
    # name_tiered), so the print of a row with no name is a function of its own (never a bare `None`).
    text, reason = _tiered_name_text(row, parsed.smiles, parsed.on_failure, not emit_tier_given)
    if parsed.verbose:
        print(f"SMILES: {parsed.smiles}\nStyle:  {parsed.style}\nName:   {text}")
    else:
        print(text)
    if reason is not None:
        print(reason, file=sys.stderr)
    if not emit_tier_given and _is_below_pin(row.get("tier")):
        print(_BELOW_PIN_NOTE, file=sys.stderr)
    return 0


def _print_confidence(result: dict) -> None:
    """The lines of ``--confidence`` for one record of ``name_compound(..., include_confidence=True)``."""
    print(f"Name:       {result['name']}")
    # C4-D: an UNMEASURED candidate reports confidence None / factors {} rather than a fabricated 1.0 (the
    # old behaviour scored a name that dropped two thirds of the molecule as perfect). Print the honest
    # verdict; never format None as a number.
    _conf = result.get('confidence')
    if _conf is None:
        print("Confidence: unverified (no coverage measurement was taken)")
    else:
        print(f"Confidence: {_conf:.4f}")
    if result.get('tier'):
        print(f"Tier:       {result['tier']}")
    print(f"Handler:    {result['handler']}")
    if result.get('factors'):
        print("Factors:")
        for k, v in result['factors'].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")


def _process_batch(input_file: str, output_file: str, style: str,
                    verbose: bool, confidence: bool = False,
                    trivial_fallback: bool = False,
                    binding_proof: str = "off",
                    on_failure: str = DEFAULT_ON_FAILURE,
                    emit_tier: str = "pin",
                    provenance: bool = False,
                    enable_triviality_controller: bool = False,
                    emit_tier_given: bool = True,
                    time_limit: Optional[float] = None) -> int:
    """Process multiple SMILES from a file.

    ``binding_proof`` is honoured here exactly as on the single-SMILES path:
    without it ``--binding-proof audit --batch f.txt`` parsed cleanly and did
    no proof work at all, which is the one failure mode an audit flag must
    not have. Validated ONCE up front rather than per row -- the per-row
    ``except Exception`` below would otherwise turn a typo'd mode into a
    silent "ERROR:" line and a zero-proof run.

    The default (``emit_tier`` the best-effort tier, ``emit_tier_given`` False), a tier named with
    ``emit_tier`` (other than ``pin``) and ``provenance`` name the lines as one SMILES is named: through one
    ``Orthonym`` (:func:`_tiered_namer`, built once for the whole file) and its ``name_tiered``. The second
    column is then the name of that tier (a line with no name follows ``on_failure`` as one SMILES does) or,
    with ``provenance``, the JSON row of ``--provenance``. For the default, one note on the error stream at
    the end of the run says that a name below ``pin_verified`` is not the PIN (once, however many lines got
    such a name, and none when no line did). ``confidence`` names the lines through ``name_compound`` and its
    record, which carries the tier for a tier above pin; with ``provenance`` it raises ValueError, once, up
    front. ``--emit-tier pin`` names the lines by ``name_compound``, as releases up to 1.0.x did.

    The output has exactly one line per input line, in input order, so line N
    of the output belongs to line N of the input: a blank input line gives a
    blank output line, and an empty input file gives an empty output. A line that
    gets no name keeps its place: ``on_failure`` ("empty" by default) says whether
    its name column is empty (the reason is printed on the error stream), the label,
    or an ``ERROR:`` row.

    ``time_limit`` (seconds, None for none: ``--time-limit``, after:func:`cli_time_limit.prepare`) is the
    wall-clock limit of each line. A line that runs out of it is a line with no name, with the code
    ``TIME_LIMIT`` (:func:`_timeout_text`), and the next line is named as in a fresh process: the limit
    puts the process-global naming state back, and the namer of the run is built anew.
    """
    from .namer import validate_binding_proof
    validate_binding_proof(binding_proof)
    tiered = _wants_tier_row(emit_tier, provenance) and not confidence
    if confidence and provenance:
        raise ValueError(_CONFIDENCE_NEEDS_PIN)
    below_pin = False
    try:
        with open(input_file, 'r') as f:
            input_lines = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return 1
    except IOError as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        return 1

    def _new_namer():
        return _tiered_namer(style, emit_tier,
                             enable_triviality_controller=enable_triviality_controller,
                             trivial_fallback=trivial_fallback, binding_proof=binding_proof)

    namer = _new_namer() if tiered else None

    results = []
    errors = 0
    named = 0
    unnamed = 0
    timed_out = 0

    for smiles in input_lines:
        if not smiles:
            results.append("")
            continue
        named += 1
        _readable(smiles)       # RDKit's reason and the reason line; the ERROR row follows
        try:
            if tiered:
                row = _tl.timed(time_limit, namer.name_tiered, smiles)
                if provenance:
                    text = _provenance_json(row)
                else:
                    text, reason = _tiered_name_text(row, smiles, on_failure, not emit_tier_given)
                    if reason is not None:
                        print(reason, file=sys.stderr)
                unnamed += not _tiered_row_is_named(row)
                below_pin = below_pin or _is_below_pin(row.get("tier"))
                results.append(f"{smiles}\t{text}")
            elif confidence:
                result = _tl.timed(time_limit, name_compound, smiles, style=style,
                                   include_confidence=True,
                                   enable_triviality_controller=enable_triviality_controller,
                                   trivial_fallback=trivial_fallback,
                                   emit_tier=emit_tier,
                                   binding_proof=binding_proof,
                                   on_failure=on_failure)
                unnamed += is_failure_name(result['name'])
                below_pin = below_pin or _is_below_pin(result.get('tier'))
                # C4-D: None means UNMEASURED, not zero — emit the token
                # 'unverified' rather than formatting None or implying 0.0000.
                _conf = result.get('confidence')
                _conf_col = ('unverified' if _conf is None
                             else f"{_conf:.4f}")
                results.append(
                    f"{smiles}\t{result['name']}\t"
                    f"{_conf_col}\t{result['handler']}"
                )
            else:
                nm = _tl.timed(time_limit, name_compound, smiles, style=style,
                               enable_triviality_controller=enable_triviality_controller,
                               trivial_fallback=trivial_fallback,
                               emit_tier=emit_tier,
                               binding_proof=binding_proof,
                               on_failure=on_failure)
                unnamed += is_failure_name(nm)
                results.append(f"{smiles}\t{nm}")
        except WallClockTimeout:
            timed_out += 1
            clear_leaked_naming_state()
            namer = _new_namer() if tiered else None
            try:
                text, reason = _timeout_text(smiles, time_limit, on_failure, provenance)
            except OrthonymLimitError as e:     # --on-failure raise: the ERROR row of any other limit
                results.append(f"{smiles}\tERROR: {e}")
                errors += 1
            else:
                if reason is not None:
                    print(reason, file=sys.stderr)
                unnamed += 1
                # --confidence has three more columns: the score of a name that was not built is unverified
                results.append(f"{smiles}\t{text}\tunverified\t{_tl.CODE}" if confidence else f"{smiles}\t{text}")
        except Exception as e:
            results.append(f"{smiles}\tERROR: {e}")
            errors += 1

    output_text = "".join(line + "\n" for line in results)
    counts = f"Processed {named} SMILES, {errors} errors, {unnamed} unnamed"
    if time_limit:      # the lines that ran out of time are among the unnamed (the errors with --on-failure raise)
        counts += f", {timed_out} timed out"

    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(output_text)
            if verbose:
                print(counts)
                print(f"Output written to: {output_file}")
        except IOError as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(output_text)
        if verbose:
            print(f"\n{counts}", file=sys.stderr)
    if below_pin and not emit_tier_given and not provenance:
        print(_BELOW_PIN_NOTE, file=sys.stderr)

    return 0 if errors == 0 else 1


def _render_node(node, indent: str = "", is_last: bool = True) -> None:
    """a phase: recursively render a NameTreeNode with indented
    box-drawing connectors. Nested ``prefixes`` subtrees render recursively
    so a multi-prefix molecule shows its full structure (not a flat list)."""
    connector = "+-" if is_last else "|-"
    print(f"{indent}{connector} parent_stem: {node.parent_stem!r}")
    child = indent + ("   " if is_last else "|  ")
    if node.locants:
        print(f"{child}|- locants: {node.locants}")
    if node.suffix:
        print(f"{child}|- suffix: {node.suffix!r}")
    if node.stereo:
        print(f"{child}|- stereo: {node.stereo!r}")
    if node.unsaturation_locants and any(node.unsaturation_locants):
        print(f"{child}|- unsaturation_locants: {node.unsaturation_locants}")
    if node.multiplicative_prefix:
        print(f"{child}|- multiplicative_prefix: {node.multiplicative_prefix!r}")
    for i, sub in enumerate(node.prefixes):
        _render_node(sub, child, is_last=(i == len(node.prefixes) - 1))


def _print_tree_text(result) -> None:
    """a phase / a phase text renderer for ``--dump-tree``.

    Renders a ``NamingResult`` as an indented, chemist-readable tree with
    recursive ``prefixes`` subtrees. a phase: every reachable handler plus
    the ion/retained boundary fallback (namer.name_with_tree) populates a tree,
    so the Phase-160 ``tree=None`` placeholder is retired.
    """
    tree = result.tree
    if tree is None:
        # Defensive only: name_with_tree's fallback guarantees a non-null
        # tree for any non-empty name; this branch is reachable only for an
        # empty name (degenerate input).
        print(f"NameTree: {result.name!r}  (tree=None)")
        return
    cite = f", cite={tree.iupac_section_cite}" if tree.iupac_section_cite else ""
    print(f"NameTree: {result.name}  (class_id={tree.class_id or 'unspecified'}{cite})")
    _render_node(tree, indent="", is_last=True)
    if result.atom_to_locant_hint is not None:
        print(f"   (atom_to_locant_hint: {result.atom_to_locant_hint!r})")


if __name__ == "__main__":
    sys.exit(main())
