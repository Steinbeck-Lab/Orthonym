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

from . import name_compound, __version__


def main(args: List[str] = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Orthonym: Generate IUPAC names from SMILES",
        prog="orthonym"
    )
    
    parser.add_argument(
        "smiles",
        nargs="?",
        help="SMILES string to convert to IUPAC name"
    )
    
    parser.add_argument(
        "--style",
        choices=["pin", "general", "cas"],
        default="pin",
        help="Naming style: pin (Preferred IUPAC Name), general, or cas (default: pin)"
    )
    
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including SMILES and style"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=str,
        help="Process multiple SMILES from a file (one per line)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for batch processing (default: stdout)"
    )

    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Show confidence metadata alongside the name"
    )

    # Phase 156 Plan-03 Task 4 (CONTEXT.md D-17 telemetry):
    # Print the OPSIN-grammar pre-validation seven-bucket counter
    # histogram to stderr after naming. Always goes through the
    # `Orthonym(...).get_validation_stats()` accessor — never reads
    # a module-global counter (AP-19).
    parser.add_argument(
        "--validation-stats",
        action="store_true",
        help="Print OPSIN grammar validation counters (D-17 telemetry) to stderr"
    )

    # Phase 158 Plan-03 Task 9 (CONTEXT.md D-16 telemetry):
    # Print the CFR class-first dispatch counter histogram to stderr
    # after naming. Default-OFF (additive; existing CLI behavior
    # unchanged). Goes through Orthonym(...).get_dispatch_stats()
    # accessor per CONTEXT D-16 + AP-6 (never reads a module-global
    # counter). Stderr-only emission keeps stdout clean for piped use.
    parser.add_argument(
        "--dispatch-stats",
        action="store_true",
        help="Print CFR dispatch counters (Phase 158 D-16 telemetry) to stderr"
    )

    # Phase 160 Plan-04 (CONTEXT D-19) + Phase 165: --dump-tree emits the
    # NameTreeNode IR for the given SMILES. Default format is "text"
    # (chemist-readable indented tree); --format json emits dataclasses.asdict()
    # JSON for machine consumption. The CLI invokes Orthonym.name_with_tree(smi)
    # which returns NamingResult(name, tree, atom_to_locant_hint). Phase 165:
    # every reachable handler plus the ion/retained boundary fallback populates
    # a tree, so the dump always shows a structured-or-coarse IR (recursive).
    parser.add_argument(
        "--dump-tree",
        action="store_true",
        help="Dump the NameTreeNode IR for the given SMILES (Phase 160 DECOMP-04)"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for --dump-tree (default: text)"
    )

    # Phase 162 ML Fallback Gate (MLF-01 + CONTEXT D-08).
    # --allow-ml-fallback: opt-in only; default OFF (MLF-01 non-negotiable).
    #   Requires the [ml] optional extra (pip install orthonym[ml]).
    # --ml-opsin-parse-required: gates the expensive P5 OPSIN-subprocess
    #   pattern in the quality-gate predicate. Default ON for production
    #   correctness; --no-ml-opsin-parse-required disables for the
    #   MLF-04 dual-config raw-attach-rate measurement mode.
    ml_group = parser.add_argument_group("ML Fallback (Phase 162)")
    ml_group.add_argument(
        "--allow-ml-fallback",
        dest="allow_ml_fallback",
        action="store_true",
        default=False,
        help=(
            "Enable opt-in ML fallback (STOUT-pypi). Default OFF; "
            "rule-based pipeline is the only path under default settings "
            "(MLF-01). Requires the [ml] optional extra: "
            "pip install orthonym[ml]. See 162-AUDIT-MLF.md § 1.2."
        ),
    )
    ml_group.add_argument(
        "--ml-opsin-parse-required",
        action="store_true",
        default=True,
        help=(
            "Quality-gate OPSIN-parse criterion (Phase 162 D-08). "
            "Default True. Use --no-ml-opsin-parse-required to disable "
            "(raw-attach-rate measurement mode; useful for MLF-04 "
            "dual-config benchmark)."
        ),
    )
    ml_group.add_argument(
        "--no-ml-opsin-parse-required",
        dest="ml_opsin_parse_required",
        action="store_false",
        help=argparse.SUPPRESS,
    )

    parsed = parser.parse_args(args)
    
    # Batch processing mode
    if parsed.batch:
        return _process_batch(parsed.batch, parsed.output, parsed.style,
                              parsed.verbose, parsed.confidence)

    # Single SMILES mode
    if not parsed.smiles:
        parser.print_help()
        return 1

    try:
        # Phase 156 Plan-03 Task 4: --validation-stats branch goes
        # through Orthonym(...).get_validation_stats() per AP-19
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

        # Phase 158 Plan-03 Task 9 + Phase 160 D-18: --dispatch-stats
        # branch goes through Orthonym(...).get_dispatch_stats() AND
        # Orthonym(...).get_inner_dispatch_stats() per CONTEXT D-16 +
        # D-18 + AP-6 (never read a module-global counter). Default-OFF;
        # stderr-only. Histogram is sorted by count descending for
        # readability; StoutClass members print by .name (uppercase
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
            # Phase 160 D-18: also print inner-dispatch counters.
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

        # Phase 160 Plan-04 (CONTEXT D-19): --dump-tree dispatches BEFORE
        # the default print(name) branch so the IR is the sole stdout
        # emission. Default format is "text" (chemist-readable indented
        # tree); --format json emits dataclasses.asdict() JSON.
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

        # Phase 162: thread parsed CLI flags to the Orthonym() constructor
        # via name_compound's new kwargs. getattr defensive pattern preserves
        # backwards compat for any direct callers that pre-date Phase 162.
        ml_kwargs = {
            "allow_ml_fallback": getattr(parsed, "allow_ml_fallback", False),
            "opsin_parse_required": getattr(parsed, "ml_opsin_parse_required", True),
        }

        if parsed.confidence:
            result = name_compound(parsed.smiles, style=parsed.style,
                                   include_confidence=True, **ml_kwargs)
            print(f"Name:       {result['name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Handler:    {result['handler']}")
            if result.get('factors'):
                print("Factors:")
                for k, v in result['factors'].items():
                    print(f"  {k}: {v:.4f}")
            if result.get('ml_fallback_used'):
                print(f"ML fallback used: True")
                print(f"ML model version: {result.get('ml_model_version', 'unknown')}")
        elif parsed.verbose:
            name = name_compound(parsed.smiles, style=parsed.style, **ml_kwargs)
            print(f"SMILES: {parsed.smiles}")
            print(f"Style:  {parsed.style}")
            print(f"Name:   {name}")
        else:
            name = name_compound(parsed.smiles, style=parsed.style, **ml_kwargs)
            print(name)

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def _process_batch(input_file: str, output_file: str, style: str,
                    verbose: bool, confidence: bool = False) -> int:
    """Process multiple SMILES from a file."""
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
                                       include_confidence=True)
                results.append(
                    f"{smiles}\t{result['name']}\t"
                    f"{result['confidence']:.4f}\t{result['handler']}"
                )
            else:
                nm = name_compound(smiles, style=style)
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
    """Phase 165 D-04: recursively render a NameTreeNode with indented
    box-drawing connectors. Nested ``prefixes[]`` subtrees render recursively
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
    """Phase 160 D-19 / Phase 165 D-04 text renderer for ``--dump-tree``.

    Renders a ``NamingResult`` as an indented, chemist-readable tree with
    recursive ``prefixes[]`` subtrees. Phase 165: every reachable handler plus
    the ion/retained boundary fallback (namer.name_with_tree) populates a tree,
    so the Phase-160 ``tree=None`` placeholder is retired.
    """
    tree = result.tree
    if tree is None:
        # Defensive only: name_with_tree's SC-3 fallback guarantees a non-null
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
