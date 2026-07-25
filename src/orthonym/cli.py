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


def _emit_tier_flags(emit_tier: str) -> dict:
    """Map a ``--emit-tier`` value to the Orthonym general-engine flag triple.

    Single source of truth for the tier -> namer-flag contract (v27 Phase 6
    T6.1). Both the CLI (``main``) and ``tests/unit/test_emit_tier_flags.py``
    read this so the invariant table cannot silently drift.

    Invariant table (0-wrong-critical):
      pin         -> gf=F, gfu=F, aag=F   (PIN-or-abstain; byte-identical default)
      valid       -> gf=T, gfu=F, aag=F   (RT-verified general-engine names)
      complete    -> gf=T, gfu=F, aag=T   (RT-verified aggressive aromatic/hetero)
      best-effort -> gf=T, gfu=T, aag=T   (complete's production PLUS the T4
                                           OPSIN-unverified opt-in)

    ``general_fallback_unverified`` (gfu) is the UNIQUE best-effort
    discriminator; ``allow_aromatic_general`` (aag) is True for BOTH complete
    and best-effort so best-effort candidate production is a superset of
    complete's (the T6.1 fix — previously aag was complete-only, making
    best-effort under-cover the P1 cage machinery).
    """
    return {
        "general_fallback": emit_tier != "pin",
        "general_fallback_unverified": emit_tier == "best-effort",
        "allow_aromatic_general": emit_tier in ("complete", "best-effort"),
    }


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

    # Phase 168 Triviality Controller (TRIV-01/02/03 + CONTEXT D-08).
    triv_group = parser.add_argument_group("Triviality Controller (Phase 168)")
    triv_group.add_argument(
        "--enable-triviality-controller",
        dest="enable_triviality_controller",
        action="store_true",
        default=False,
        help=(
            "Enable the triviality controller (Phase 168). Default OFF "
            "(Stage A SACRED canary invariant). When True, the controller "
            "swaps systematic PIN-eligible parents (benzene/phenol/aniline/"
            "benzoic acid/etc.) to retained PIN forms at the name-tree IR "
            "layer per IUPAC P-15.1.8.1..3. See 168-CONTEXT.md D-08. Env "
            "override: ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1."
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
            "Fall back to a general-only (non-PIN) retained trivial name when "
            "the default PIN pipeline cannot derive a preferred IUPAC name. "
            "Never downgrades a derivable PIN (e.g. glycerol SMILES still "
            "yields propane-1,2,3-triol). Default OFF (PIN fails closed)."
        ),
    )

    # v25 G3: confidence-tiered output surface. Default 'pin' is the
    # existing PIN-or-abstain behavior byte-identically; 'valid' adds
    # RT-verified general-engine names (T3); 'best-effort' additionally
    # ships E1-certified names OPSIN could not verify (T4). No tier ever
    # ships a name OPSIN parsed to a DIFFERENT structure.
    parser.add_argument(
        "--emit-tier",
        dest="emit_tier",
        choices=["pin", "valid", "complete", "best-effort"],
        default="pin",
        help=(
            "Output tier: pin (default, PIN-or-abstain), valid "
            "(adds RT-verified general-engine names), complete (v26; adds "
            "RT-verified aggressive general aromatic/heterocyclic "
            "fallbacks; non-PIN allowed), best-effort (adds E1-certified "
            "but OPSIN-unverified names)."
        ),
    )
    parser.add_argument(
        "--provenance",
        action="store_true",
        default=False,
        help=(
            "Emit {name,tier,is_pin,source,opsin,gates_passed} JSON "
            "instead of the bare name (v25 G3)."
        ),
    )
    parser.add_argument(
        "--engine-only",
        dest="engine_only",
        action="store_true",
        default=False,
        help=(
            "DIAGNOSTIC (v25): bypass the PIN path and print the general "
            "engine's raw emission (E1-audited, NOT OPSIN-RT-gated) or its "
            "refusal reason. Never a PIN; for inspection/comparison only."
        ),
    )

    parser.add_argument(
        "--binding-proof",
        dest="binding_proof",
        choices=["off", "audit", "enforce"],
        default="off",
        help=(
            "v29 P1 name<->graph binding proof: off (default, no proof work "
            "at all), audit (re-assert the producer's binding spine against "
            "the FINAL returned name and log the findings; the name is never "
            "changed), enforce (additionally abstain when that proof fails). "
            "Applies to the tiered emit surface."
        ),
    )

    parsed = parser.parse_args(args)

    # Batch processing mode
    if parsed.batch:
        return _process_batch(parsed.batch, parsed.output, parsed.style,
                              parsed.verbose, parsed.confidence,
                              getattr(parsed, 'trivial_fallback', False))

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

        # Phase 168 D-08: thread the triviality-controller flag through
        # name_compound. getattr defensive pattern preserves backwards compat.
        # Task 1.9: also thread the --trivial fallback flag.
        name_kwargs = {
            "enable_triviality_controller": getattr(
                parsed, "enable_triviality_controller", False),
            "trivial_fallback": getattr(parsed, "trivial_fallback", False),
        }

        # v25: --engine-only diagnostic branch (before the tier branch: a
        # pure inspection surface, never a production emit path).
        if getattr(parsed, "engine_only", False):
            import json as _json
            from rdkit import Chem as _Chem
            from orthonym.namer import Orthonym
            from orthonym.assembly.general_engine import name_general
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

        # v25 G3: tiered-output branch. Constructs the namer directly (the
        # tier flags are namer-level), prints JSON with --provenance or the
        # bare name otherwise. Dispatches before the legacy branches so the
        # default (--emit-tier pin, no --provenance) path below stays
        # byte-identical.
        _emit_tier = getattr(parsed, "emit_tier", "pin")
        # v29 P1: --binding-proof also selects this branch, because it is the
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
                binding_proof=_binding_proof,
            )
            row = namer.name_tiered(parsed.smiles)
            if parsed.provenance:
                print(_json.dumps(row))
            elif row["name"]:
                print(row["name"])
            else:
                # v28 Composer1 Task 5 fix: a clean T5 abstain has
                # name=None by design (honest "no name" contract — see
                # name_tiered's v28 Composer1 Task 5 comment). Printing
                # bare `None` to stdout would be confusing/broken output
                # for this documented flag; print a labeled abstention
                # indicator instead. Never fabricates a name.
                print(f"(no name — {row['limit_code'] or row['tier']})")
            return 0

        if parsed.confidence:
            result = name_compound(parsed.smiles, style=parsed.style,
                                   include_confidence=True, **name_kwargs)
            print(f"Name:       {result['name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Handler:    {result['handler']}")
            if result.get('factors'):
                print("Factors:")
                for k, v in result['factors'].items():
                    print(f"  {k}: {v:.4f}")
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
                    trivial_fallback: bool = False) -> int:
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
                                       include_confidence=True,
                                       trivial_fallback=trivial_fallback)
                results.append(
                    f"{smiles}\t{result['name']}\t"
                    f"{result['confidence']:.4f}\t{result['handler']}"
                )
            else:
                nm = name_compound(smiles, style=style,
                                   trivial_fallback=trivial_fallback)
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
