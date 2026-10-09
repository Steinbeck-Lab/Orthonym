"""
Command-line interface for Orthonym.

Usage:
    orthonym "CCO"
    orthonym "CC(=O)O" --style general
    python -m orthonym "SMILES"
"""

import argparse
import sys
from typing import List

from . import __version__, name_compound


def _emit_tier_flags(emit_tier: str) -> dict:
    """Map a ``--emit-tier`` value to the Orthonym general-engine flag triple.

    Single source of truth for the tier -> namer-flag contract (a phase
    T6.1). Both the CLI (``main``) and ``tests/unit/test_emit_tier_flags.py``
    read this so the invariant table cannot silently drift.

    Invariant table (0-wrong-critical):
      pin -> gf=F, gfu=F, aag=F (PIN-or-abstain; byte-identical default)
      valid -> gf=T, gfu=F, aag=F (RT-verified general-engine names)
      complete -> gf=T, gfu=F, aag=T (RT-verified aggressive aromatic/hetero)
      best-effort -> gf=T, gfu=T, aag=T (complete's production PLUS the T4
                                           OPSIN-unverified opt-in)

    ``general_fallback_unverified`` (gfu) is the UNIQUE best-effort
    discriminator; ``allow_aromatic_general`` (aag) is True for BOTH complete
    and best-effort so best-effort candidate production is a superset of
    complete's (the T6.1 fix — previously aag was complete-only, making
    best-effort under-cover the P1 cage machinery).

     adds a fifth tier, ``full-coverage`` (the opt-in flag tier that
    arms the D2 general coordination-additive namer):

      full-coverage -> gf=T, gfu=T, aag=T, full_coverage=T

    It is best-effort's production superset (identical gf/gfu/aag triple) PLUS
    its OWN ``full_coverage`` marker bit -- deliberately NOT aliased to
    best-effort's triple, so D2's dispatch point downstream can distinguish
    "best-effort tier" from "full-coverage tier". The ``full_coverage`` key is
    present (``False``) on every other tier, so a consumer can always read it.
    With the flag off (every non-full-coverage tier) D2 is never reached and
    output is byte-identical -- the SP5.4 isolation property.
    """
    _be_or_fc = emit_tier in ("best-effort", "full-coverage")
    return {
        "general_fallback": emit_tier != "pin",
        "general_fallback_unverified": _be_or_fc,
        "allow_aromatic_general": emit_tier in (
            "complete", "best-effort", "full-coverage"),
        "full_coverage": emit_tier == "full-coverage",
    }


def main(args: List[str] = None) -> int:
    """Main entry point for CLI."""
    # Before anything can start the JVM: a native crash must print the Python
    # stack, not only a JVM hs_err file (orthonym.diagnostics, audit 2026-09-03).
    from orthonym.diagnostics import enable_crash_traceback
    enable_crash_traceback()
    parser = argparse.ArgumentParser(
        description=("Orthonym: IUPAC names for chemical structures. Give a SMILES string "
                     "and get its IUPAC name, checked by reading it back with OPSIN, or a "
                     "label that says why no name was given. The few names OPSIN cannot "
                     "read in full are marked by --provenance."),
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
              "acid dihydrate') and the axial descriptors Ra and Sa. The default "
              "tier's PIN-only rule applies to pin only, so with general or cas a "
              "name that pin declines with NO_VERIFIED_PIN is returned. cas is "
              "accepted and at present gives the names of general, except that "
              "hydrate names and axial descriptors keep their pin form.")
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help=("Also print the SMILES and the style. With --batch and --output, also "
              "say how many lines were named and how many failed.")
    )

    parser.add_argument(
        "--batch", "-b",
        type=str,
        help=("Name every SMILES in this file, one per line, and print one "
              "'SMILES<tab>name' line each. --emit-tier and --provenance do not apply"
              " to batch runs yet.")
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="With --batch: write the lines to this file instead of the screen."
    )

    parser.add_argument(
        "--confidence",
        action="store_true",
        help=("Print the name with a coverage score, the part of the engine that "
              "built it, and the parts of the score.")
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
              "pre-check passed or repaired a candidate name.")
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
              "the engine took.")
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
        help="Print the parts of the name as a tree instead of the name."
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Layout for --dump-tree: text (an indented tree; the default) or json."
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
            " tiers, and the default tier declines them: names from a small "
            "last-resort table, and the trivial natural-product names of molecules "
            "whose preferred bridged fused name the engine does not build yet "
            "(diamorphine). --provenance labels both systematic_verified, source "
            "trivial_retained. A natural-product name built on a parent, such as "
            "'(9R,13S,14S)-3-methoxy-17-methylmorphinan', is not a trivial name, "
            "and --trivial does not return it at the default tier."
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
        choices=["pin", "valid", "complete", "best-effort", "full-coverage"],
        default="pin",
        help=(
            "Which names to return. pin (the default): a name only when the "
            "pipeline can build the preferred IUPAC name (PIN), that is when the "
            "strict PIN path built the name and verified it (tier pin_verified). "
            "The exceptions are names from the natural-product and "
            "metal-complex lists, the name formats absent from OPSIN's grammar "
            "(for example inositols, phanes and thioperoxols), PINs whose "
            "stereodescriptors OPSIN cannot read, for which the default tier "
            "compares the constitution, and, with --trivial, a retained trivial "
            "name. Otherwise it declines, with the reason "
            "code NO_VERIFIED_PIN when it built a name that is not a verified PIN. "
            "The rule applies to the default --style pin. "
            "valid: also names from the general engine. complete: also general "
            "names for aromatic and heterocyclic ring systems. best-effort: also "
            "the last-resort producers, von Baeyer and spiro names for ring systems "
            "of up to 100 skeletal atoms and 11 rings (the other tiers build them up to "
            "40 atoms and 8 rings), and adducts with a one-atom ion. full-coverage: also the coordination-name "
            "builder for metal tetrapyrrole and corrin complexes, which builds a "
            "name or declines. The wider tiers also return the names the default "
            "tier declines, each labelled with its tier; there every name must "
            "pass a full-InChIKey OPSIN round trip, except a name from the "
            "natural-product and metal-complex lists, so the name formats absent "
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
            "prefix_order_fallback, verified and spelling_failures. "
            "'verified' is opsin (OPSIN read the whole name back to the same "
            "molecule), opsin_constitution (the same constitution; the "
            "stereodescriptors were not confirmed by OPSIN), identity (a name from "
            "an exact-match list: a metal-complex name found by the input's exact "
            "InChIKey, or a natural-product parent name found by its exact "
            "structure; OPSIN cannot read these names) or unverified (no "
            "read-back recorded). 'gate_outcome' says what the final OPSIN check "
            "did, for example self_consistency_verified, suppressed, not_run or "
            "carveout:<class>. 'spelling_failures' lists the spelling rules of the "
            "PIN (rule id and detail) that a name in PIN form breaks, and the rule "
            "of a recorded non-PIN part that lowered the name; such a name is "
            "not labelled pin_verified, and pin declines it. One SMILES at a time; "
            "not with --batch."
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
            " that it declined. Never a preferred name; do not use it as a name."
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

    parsed = parser.parse_args(args)

    # Nothing to name: the help needs no jars, so print it before the jar check (which
    # downloads a missing jar and makes a first call without arguments slow).
    if not parsed.smiles and not parsed.batch and not parsed.fetch_jars:
        parser.print_help()
        return 1

    from orthonym.jars import JarUnavailable, fetch_all, require_all
    if parsed.fetch_jars:
        try:
            fetch_all(verbose=True)
        except JarUnavailable as exc:
            print(f"orthonym: {exc}", file=sys.stderr)
            return 1
        return 0
    try:
        require_all()
    except JarUnavailable as exc:
        print(f"orthonym: {exc}\n"
              "orthonym: to name without the jars (no OPSIN check, RDKit stereo labels), "
              "set ORTHONYM_ALLOW_REDUCED=1", file=sys.stderr)
        return 2

    # Batch processing mode
    if parsed.batch:
        return _process_batch(parsed.batch, parsed.output, parsed.style,
                              parsed.verbose, parsed.confidence,
                              getattr(parsed, 'trivial_fallback', False),
                              getattr(parsed, 'binding_proof', 'off'))

    # Single SMILES mode
    if not parsed.smiles:
        parser.print_help()
        return 1

    try:
        # a phase Plan-03 Task 4: --validation-stats branch goes
        # through Orthonym(...).get_validation_stats per
        # (never read a module-global counter). Branch is taken BEFORE
        # the existing default `print(name)` so the histogram is the
        # only stderr output path when the flag is on.
        if parsed.validation_stats:
            from orthonym.namer import Orthonym
            namer = Orthonym(style=parsed.style)
            name = namer.name(parsed.smiles)
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
            namer = Orthonym(style=parsed.style)
            name = namer.name(parsed.smiles)
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
            namer = Orthonym(style=parsed.style)
            result = namer.name_with_tree(parsed.smiles)
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
        }

        #: --engine-only diagnostic branch (before the tier branch: a
        # pure inspection surface, never a production emit path).
        if getattr(parsed, "engine_only", False):
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

        #: tiered-output branch. Constructs the namer directly (the
        # tier flags are namer-level), prints JSON with --provenance or the
        # bare name otherwise. Dispatches before the legacy branches so the
        # default (--emit-tier pin, no --provenance) path below stays
        # byte-identical.
        _emit_tier = getattr(parsed, "emit_tier", "pin")
        #: --binding-proof also selects this branch, because it is the
        # only single-SMILES surface that constructs the namer directly (the
        # default path goes through name_compound). Without it the flag would
        # parse and then silently do nothing -- the same failure mode the
        # ValueError in Orthonym.__init__ exists to prevent.
        _binding_proof = getattr(parsed, "binding_proof", "off")
        if _emit_tier != "pin" or parsed.provenance or _binding_proof != "off":
            import json as _json

            from orthonym.namer import Orthonym
            _tier_flags = _emit_tier_flags(_emit_tier)
            namer = Orthonym(
                style=parsed.style,
                enable_triviality_controller=name_kwargs[
                    "enable_triviality_controller"],
                trivial_fallback=name_kwargs["trivial_fallback"],
                general_fallback=_tier_flags["general_fallback"],
                general_fallback_unverified=_tier_flags[
                    "general_fallback_unverified"],
                allow_aromatic_general=_tier_flags["allow_aromatic_general"],
                full_coverage=_tier_flags.get("full_coverage", False),
                binding_proof=_binding_proof,
            )
            row = namer.name_tiered(parsed.smiles)
            if parsed.provenance:
                print(_json.dumps(row))
            elif row["name"]:
                print(row["name"])
            else:
                # Composer1 Task 5 fix: a clean abstain has
                # name=None by design (honest "no name" contract — see
                # name_tiered's Composer1 Task 5 comment). Printing
                # bare `None` to stdout would be confusing/broken output
                # for this documented flag; print a labeled abstention
                # indicator instead. Never fabricates a name.
                print(f"(no name — {row['limit_code'] or row['tier']})")
            return 0

        if parsed.confidence:
            result = name_compound(parsed.smiles, style=parsed.style,
                                   include_confidence=True, **name_kwargs)
            print(f"Name:       {result['name']}")
            # C4-D: an UNMEASURED candidate reports confidence None /
            # factors {} rather than a fabricated 1.0 (the old behaviour scored
            # a name that dropped two thirds of the molecule as perfect). Print
            # the honest verdict; never format None as a number.
            _conf = result.get('confidence')
            if _conf is None:
                print("Confidence: unverified (no coverage measurement was taken)")
            else:
                print(f"Confidence: {_conf:.4f}")
            print(f"Handler:    {result['handler']}")
            if result.get('factors'):
                print("Factors:")
                for k, v in result['factors'].items():
                    print(f"  {k}: {v:.4f}" if isinstance(v, (int, float))
                          else f"  {k}: {v}")
        elif parsed.verbose:
            name = name_compound(parsed.smiles, style=parsed.style, **name_kwargs)
            print(f"SMILES: {parsed.smiles}")
            print(f"Style:  {parsed.style}")
            print(f"Name:   {name}")
        else:
            name = name_compound(parsed.smiles, style=parsed.style, **name_kwargs)
            print(name)

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def _process_batch(input_file: str, output_file: str, style: str,
                    verbose: bool, confidence: bool = False,
                    trivial_fallback: bool = False,
                    binding_proof: str = "off") -> int:
    """Process multiple SMILES from a file.

    ``binding_proof`` is honoured here exactly as on the single-SMILES path:
    without it ``--binding-proof audit --batch f.txt`` parsed cleanly and did
    no proof work at all, which is the one failure mode an audit flag must
    not have. Validated ONCE up front rather than per row -- the per-row
    ``except Exception`` below would otherwise turn a typo'd mode into a
    silent "ERROR:" line and a zero-proof run.
    """
    from .namer import validate_binding_proof
    validate_binding_proof(binding_proof)
    try:
        with open(input_file, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return 1
    except IOError as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        return 1

    results = []
    errors = 0

    for smiles in smiles_list:
        try:
            if confidence:
                result = name_compound(smiles, style=style,
                                       include_confidence=True,
                                       trivial_fallback=trivial_fallback,
                                       binding_proof=binding_proof)
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
                nm = name_compound(smiles, style=style,
                                   trivial_fallback=trivial_fallback,
                                   binding_proof=binding_proof)
                results.append(f"{smiles}\t{nm}")
        except Exception as e:
            results.append(f"{smiles}\tERROR: {e}")
            errors += 1

    # Write output
    output_text = "\n".join(results)

    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(output_text + "\n")
            if verbose:
                print(f"Processed {len(smiles_list)} SMILES, {errors} errors")
                print(f"Output written to: {output_file}")
        except IOError as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            return 1
    else:
        print(output_text)
        if verbose:
            print(f"\nProcessed {len(smiles_list)} SMILES, {errors} errors",
                  file=sys.stderr)

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
